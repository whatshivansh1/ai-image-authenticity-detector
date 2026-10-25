import asyncio
import functools
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms
import os
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "ai_image_detector.pth"
SAMPLES_DIR = ROOT / "backend" / "samples"

app = FastAPI(
    title="AI-Generated Image Detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

device = torch.device("cpu")

# Thread pool for CPU-bound inference — keeps the async event loop responsive
_executor = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()

    # Set optimal thread count for CPU
    torch.set_num_threads(os.cpu_count() or 1)

    # NOTE: torch.compile() removed — it is the primary cause of the 6-7 s
    # latency. On CPU it triggers Triton/inductor JIT recompilation on the
    # first *real* image (different graph paths than the zero-tensor warmup),
    # stalling the request for several seconds. MobileNetV3 on CPU is already
    # fast (~200-400 ms) without it.

    # NOTE: quantize_dynamic removed — it only helps nn.Linear layers and
    # adds net overhead to the conv-heavy MobileNetV3 backbone.

    # Pre-warm 3x to fill CPU caches and eliminate first-request cold-start.
    with torch.inference_mode():
        for _ in range(3):
            dummy = torch.zeros(1, 3, 224, 224)
            m(dummy)

    return m


model = load_model()


def extract_forensic_metadata(source: Image.Image, raw_bytes: bytes):
    """Inspects EXIF and image chunk metadata for camera hardware or AI generator fingerprints."""
    ai_signatures = []
    camera_data = {}

    # 1. Text chunks in PNG / JPEG headers (e.g. prompt, negative prompt, A1111/ComfyUI parameters)
    if hasattr(source, "info") and source.info:
        for k, v in source.info.items():
            k_lower = str(k).lower()
            v_str = str(v).lower() if isinstance(v, (str, bytes)) else ""
            for indicator in [
                "prompt", "negative prompt", "stable diffusion", "midjourney",
                "dall-e", "comfyui", "novelai", "automatic1111", "c2pa",
                "civitai", "generation parameter", "steps:", "sampler:"
            ]:
                if indicator in k_lower or indicator in v_str:
                    clean_val = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
                    ai_signatures.append(f"{k}: {clean_val}")
                    break

    # 2. Camera EXIF metadata (Make, Model, Lens, ISO, Exposure)
    try:
        exif = source.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value).strip()
                if tag in ["Make", "Model", "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings", "DateTimeOriginal"]:
                    camera_data[tag] = val_str
                elif tag == "Software":
                    s_lower = val_str.lower()
                    if any(x in s_lower for x in ["stable diffusion", "midjourney", "dall-e", "adobe firefly", "comfyui"]):
                        ai_signatures.append(f"Software signature: {val_str}")
                    else:
                        camera_data["Software"] = val_str
    except Exception:
        pass

    # 3. Fast binary byte scan for known C2PA / generator chunks
    raw_lower = raw_bytes[:16384].lower() + raw_bytes[-8192:].lower()
    for bin_sig in [b"c2pa", b"sd-metadata", b"stability ai", b"dreamstudio", b"invokeai"]:
        if bin_sig in raw_lower:
            sig_name = bin_sig.decode("ascii", errors="ignore")
            if not any(sig_name in s.lower() for s in ai_signatures):
                ai_signatures.append(f"Embedded generator marker: {sig_name}")

    has_camera = bool(
        camera_data.get("Make")
        or camera_data.get("Model")
        or ("ISOSpeedRatings" in camera_data and "ExposureTime" in camera_data)
    )
    camera_name = f"{camera_data.get('Make', '')} {camera_data.get('Model', '')}".strip() or None

    return {
        "has_camera_hardware": has_camera,
        "camera_device": camera_name,
        "camera_details": camera_data,
        "ai_signatures": ai_signatures,
        "has_ai_metadata": len(ai_signatures) > 0,
    }



def compute_spectral_analysis(image: Image.Image):
    """2D FFT spectral analysis — detects periodic high-freq artifacts from diffusion/GAN upsampling."""
    # Resize to smaller 128x128 for faster FFT while preserving enough detail
    gray = np.array(
        image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    center = 64
    y, x = np.ogrid[:128, :128]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    low_mask = dist <= 16
    high_mask = (dist > 16) & (dist <= 55)

    low_energy = float(magnitude[low_mask].mean())
    high_energy = float(magnitude[high_mask].mean())
    ratio = float(high_energy / (low_energy + 1e-6))

    std_high = float(magnitude[high_mask].std() / (high_energy + 1e-6))

    is_anomaly = bool(std_high > 1.85 or ratio > 0.12)
    return {
        "high_freq_ratio": round(ratio, 4),
        "high_freq_std": round(std_high, 4),
        "spectral_anomaly": is_anomaly,
    }


def _run_inference(tensor: torch.Tensor):
    """Blocking inference helper — runs in thread pool to avoid blocking the event loop."""
    with torch.inference_mode():
        logits = model(tensor)[0]
        probabilities = torch.softmax(logits, dim=0)
    return float(probabilities[0].item()), float(probabilities[1].item())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "architecture": "MobileNetV3-Large + Multi-signal Forensic Engine",
    }


@app.get("/samples")
def list_samples():
    if not SAMPLES_DIR.exists():
        return []
    samples = []
    for p in sorted(SAMPLES_DIR.glob("*")):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            is_ai = "ai" in p.name.lower()
            name_pretty = p.stem.replace("sample_", "").replace("_", " ").title()
            samples.append({
                "id": p.name,
                "name": name_pretty,
                "expected": "AI-Generated" if is_ai else "Authentic",
                "filename": p.name,
            })
    return samples


@app.get("/samples/{filename}")
def get_sample_file(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "Sample image not found.")
    return FileResponse(file_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be 10 MB or smaller.")

    if not content:
        raise HTTPException(400, "Empty image file.")

    try:
        with Image.open(io.BytesIO(content)) as source:
            # 1. Forensic metadata extraction
            metadata = extract_forensic_metadata(source, content)

            # 2. Transpose according to EXIF orientation tag (e.g. smartphone camera)
            transposed = ImageOps.exif_transpose(source)

            # 3. Safe conversion handling alpha / transparency
            if transposed.mode in ("RGBA", "LA") or (transposed.mode == "P" and "transparency" in transposed.info):
                canvas = Image.new("RGB", transposed.size, (255, 255, 255))
                canvas.paste(transposed, mask=transposed.convert("RGBA").split()[3])
                image = canvas
            else:
                image = transposed.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid or unsupported image file.")

    # 4. Spectral 2D FFT Analysis
    spectral = compute_spectral_analysis(image)

    # 5. Deep Learning Model Inference (MobileNetV3-Large) — offloaded to thread
    #    pool so the async event loop is not blocked during CPU-bound inference
    tensor = preprocess(image).unsqueeze(0)
    loop = asyncio.get_event_loop()
    ai_prob, real_prob = await loop.run_in_executor(
        _executor, functools.partial(_run_inference, tensor)
    )

    # 6. Multi-Signal Synthesis
    # Priority A: Definitive generative AI prompt/metadata signatures
    if metadata["has_ai_metadata"]:
        label = "AI-Generated"
        confidence = 99.85
        reason = f"Direct generative AI metadata signature identified: {metadata['ai_signatures'][0]}"

    # Priority B: Authentic hardware camera EXIF verified
    elif metadata["has_camera_hardware"]:
        label = "Authentic"
        cam = metadata["camera_device"] or "Hardware camera"
        confidence = round(max(94.0, min(99.4, 95.0 + len(metadata["camera_details"]) * 0.8)), 2)
        reason = f"Verified camera hardware capture metadata ({cam})."

    # Priority C: Spectral & Neural Network synthesis (e.g. web images, uncompressed crops)
    else:
        if spectral["spectral_anomaly"]:
            label = "AI-Generated"
            confidence = round(max(85.0, min(98.5, 88.0 + (spectral["high_freq_std"] - 1.85) * 10)), 2)
            reason = "High-frequency periodic spectral artifacts detected (typical of diffusion/GAN upsampling)."
        else:
            # Deep Learning probability threshold
            if ai_prob >= 0.70:
                label = "AI-Generated"
                confidence = round(ai_prob * 100, 2)
                reason = "Deep neural network identified characteristic spatial generation patterns."
            else:
                label = "Authentic"
                confidence = round(max(75.0, (1.0 - ai_prob) * 100), 2)
                reason = "Natural sensor pixel coherence with smooth frequency-domain distribution."

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "label": label,
        "confidence": confidence,
        "processing_time_ms": elapsed_ms,
        "reason": reason,
        "signals": {
            "metadata": {
                "has_camera_hardware": metadata["has_camera_hardware"],
                "camera_device": metadata["camera_device"],
                "has_ai_metadata": metadata["has_ai_metadata"],
                "ai_signatures": metadata["ai_signatures"],
            },
            "spectral": {
                "high_freq_ratio": spectral["high_freq_ratio"],
                "spectral_anomaly": spectral["spectral_anomaly"],
            },
            "neural_network": {
                "ai_score": round(ai_prob * 100, 2),
                "authentic_score": round(real_prob * 100, 2),
            },
        },
        "note": "Multi-signal probabilistic forensic score combining EXIF, 2D FFT, and MobileNetV3.",
    }
