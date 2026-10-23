import { useEffect, useRef, useState } from 'react'
import {
  Upload, Camera, Image as ImageIcon, ShieldCheck,
  ScanSearch, LoaderCircle, X, CheckCircle2,
  AlertCircle, Clock3, RotateCcw, Cpu, Sparkles,
  Activity, Layers, Info
} from 'lucide-react'
import './index.css'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [samples, setSamples] = useState([])
  const [dragActive, setDragActive] = useState(false)

  const fileInput = useRef(null)
  const cameraInput = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  // Load sample images list on mount
  useEffect(() => {
    fetch(`${API_URL}/samples`)
      .then(res => res.ok ? res.json() : [])
      .then(data => setSamples(data))
      .catch(() => setSamples([]))
  }, [])

  // Clipboard paste support
  useEffect(() => {
    function handlePaste(e) {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const blob = item.getAsFile()
          if (blob) chooseFile(blob)
          break
        }
      }
    }
    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [])

  function chooseFile(nextFile) {
    if (!nextFile) return
    if (!nextFile.type.startsWith('image/')) {
      setError('Please select a valid image file (JPEG, PNG, WEBP, BMP).')
      return
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError('Image must be smaller than 10 MB.')
      return
    }
    setFile(nextFile)
    setPreview(URL.createObjectURL(nextFile))
    setResult(null)
    setError('')
  }

  async function loadSample(sample) {
    try {
      setLoading(true)
      setError('')
      const res = await fetch(`${API_URL}/samples/${sample.filename}`)
      if (!res.ok) throw new Error('Could not load sample file.')
      const blob = await res.blob()
      const sampleFile = new File([blob], sample.filename, { type: blob.type || 'image/jpeg' })
      chooseFile(sampleFile)
      // Auto analyze sample
      await runAnalysis(sampleFile)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview('')
    setResult(null)
    setError('')
    if (fileInput.current) fileInput.current.value = ''
    if (cameraInput.current) cameraInput.current.value = ''
  }

  async function runAnalysis(fileToAnalyze) {
    const target = fileToAnalyze || file
    if (!target) return
    setLoading(true)
    setError('')
    setResult(null)
    const body = new FormData()
    body.append('file', target)
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.')
      setResult(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot reach the local backend. Ensure FastAPI is running on port 8000.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  async function openCamera() {
    setCameraError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is unavailable in this browser. Please use Chrome or Safari on localhost.')
      return
    }

    let devices = []
    try {
      devices = await navigator.mediaDevices.enumerateDevices()
    } catch (e) {
      console.warn('Could not enumerate devices:', e)
    }

    const videoDevices = devices.filter(d => d.kind === 'videoinput')
    const attempts = []

    if (videoDevices.length > 0 && videoDevices[0].deviceId) {
      attempts.push({ video: { deviceId: { exact: videoDevices[0].deviceId } }, audio: false })
    }
    attempts.push(
      { video: true, audio: false },
      { video: { facingMode: 'user' }, audio: false },
      { video: { facingMode: { ideal: 'environment' } }, audio: false }
    )

    let stream = null
    let lastError = null

    for (const constraints of attempts) {
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints)
        if (stream) break
      } catch (err) {
        lastError = err
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          break
        }
      }
    }

    if (!stream) {
      if (lastError?.name === 'NotAllowedError' || lastError?.name === 'PermissionDeniedError') {
        setCameraError(
          'Camera access blocked. In Chrome: click the tune/sliders icon next to localhost:5173 and allow Camera. On macOS: open System Settings → Privacy & Security → Camera → toggle Google Chrome ON.'
        )
      } else if (lastError?.name === 'NotFoundError' || lastError?.name === 'DevicesNotFoundError') {
        setCameraError(
          'macOS is hiding the camera from Chrome. Go to Apple  menu → System Settings → Privacy & Security → Camera and make sure "Google Chrome" is toggled ON. Also ensure your MacBook lid is open.'
        )
      } else if (lastError?.name === 'NotReadableError') {
        setCameraError('Camera is currently in use by another app (FaceTime, Zoom, Photo Booth). Please quit other apps and try again.')
      } else {
        setCameraError(`Camera could not start (${lastError?.name || 'Error'}): ${lastError?.message || 'Access failed'}`)
      }
      return
    }

    streamRef.current = stream
    setCameraOpen(true)
    setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(e => console.warn('Play interrupted:', e))
        }
      }
    }, 100)
  }

  function closeCamera() {
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    setCameraOpen(false)
  }

  function capturePhoto() {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob(blob => {
      if (blob) {
        const photo = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
        chooseFile(photo)
        closeCamera()
      }
    }, 'image/jpeg', 0.95)
  }

  function handleDrag(e) {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      chooseFile(e.dataTransfer.files[0])
    }
  }

  const isAI = result?.label === 'AI-Generated'

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-icon"><ScanSearch size={22} /></span>
          <span>VERITY<span className="brand-dot">.</span></span>
        </a>
        <div className="top-status"><span /> MULTI-SIGNAL FORENSIC ENGINE</div>
      </header>

      <section className="hero">
        <div className="eyebrow"><ShieldCheck size={15} /> MULTI-SIGNAL AUTHENTICITY INTELLIGENCE</div>
        <h1>See beyond<br />the <span>pixels.</span></h1>
        <p className="hero-copy">
          Forensic AI detection combining MobileNetV3 deep learning, 2D FFT spectral artifact analysis,
          and hardware EXIF sensor metadata.
        </p>
      </section>

      {/* Preset Samples Quick-Bar */}
      {samples.length > 0 && (
        <section className="samples-bar">
          <span className="samples-label"><Sparkles size={14} /> Quick Test Presets:</span>
          <div className="samples-list">
            {samples.map(s => (
              <button
                key={s.id}
                className={`sample-pill ${s.expected === 'AI-Generated' ? 'pill-ai' : 'pill-real'}`}
                onClick={() => loadSample(s)}
                disabled={loading}
              >
                <span className="pill-dot" />
                {s.name}
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="workspace">
        {/* Upload Panel */}
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div><span className="step-number">01</span><h2>Upload image</h2></div>
            {file && <button className="icon-button" onClick={reset} title="Clear image"><X size={18} /></button>}
          </div>

          {!preview ? (
            <button
              className={`dropzone ${dragActive ? 'dropzone-active' : ''}`}
              onClick={() => fileInput.current?.click()}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <span className="upload-symbol"><Upload size={26} /></span>
              <span className="drop-title">Drop your image here</span>
              <span className="drop-subtitle">or click to browse your files · Paste from clipboard supported</span>
              <span className="format-pill">JPG · PNG · WEBP · BMP · MAX 10 MB</span>
            </button>
          ) : (
            <div className="preview-wrap">
              <img className="preview-image" src={preview} alt="Selected image preview" />
              <div className="preview-tag"><ImageIcon size={14} /> {file?.name}</div>
            </div>
          )}

          <input
            ref={fileInput}
            className="hidden-input"
            type="file"
            accept="image/jpeg,image/png,image/webp,image/bmp"
            onChange={e => chooseFile(e.target.files?.[0])}
          />
          <input
            ref={cameraInput}
            className="hidden-input"
            type="file"
            accept="image/*"
            capture="environment"
            onChange={e => chooseFile(e.target.files?.[0])}
          />

          <div className="input-actions">
            <button className="secondary-button" onClick={() => fileInput.current?.click()} disabled={loading}>
              <ImageIcon size={17} /> Browse files
            </button>
            <button className="secondary-button" onClick={openCamera} disabled={loading}>
              <Camera size={17} /> Open camera
            </button>
          </div>

          {cameraError && (
            <div className="camera-error-box">
              <p className="error-message"><AlertCircle size={16} />{cameraError}</p>
              <button
                type="button"
                className="secondary-button camera-fallback-btn"
                onClick={() => cameraInput.current?.click()}
              >
                <Camera size={15} /> Open OS camera / photo capture
              </button>
            </div>
          )}

          <div className="privacy-note">
            <ShieldCheck size={16} />
            <span>Analyzed privately in-memory via local FastAPI server (&lt; 100ms SLA).</span>
          </div>

          <button className="analyze-button" disabled={!file || loading} onClick={() => runAnalysis()}>
            {loading ? (
              <><LoaderCircle className="spin" size={19} /> Performing forensic analysis...</>
            ) : (
              <><ScanSearch size={19} /> Analyze image <span>↗</span></>
            )}
          </button>
        </div>

        {/* Results Panel */}
        <div className="panel result-panel">
          <div className="panel-heading">
            <div><span className="step-number">02</span><h2>Forensic result</h2></div>
            <span className="live-label"><span /> ACTIVE</span>
          </div>

          {!result && !loading && !error && (
            <div className="empty-result">
              <div className="radar"><ScanSearch size={34} /></div>
              <h3>Awaiting analysis</h3>
              <p>Upload an image or click a quick-test preset to inspect spatial textures, frequency spectrum, and sensor metadata.</p>
              <div className="empty-stats">
                <div><span>MODEL</span><strong>MobileNetV3</strong></div>
                <div><span>FREQUENCY</span><strong>2D FFT Spectrum</strong></div>
                <div><span>HARDWARE</span><strong>EXIF Validation</strong></div>
              </div>
            </div>
          )}

          {loading && (
            <div className="loading-result">
              <LoaderCircle className="spin large-spin" size={42} />
              <h3>Analyzing forensic signals</h3>
              <p>Evaluating spatial diffusion textures, spectral upsampling harmonics, and camera sensor headers...</p>
            </div>
          )}

          {error && (
            <div className="error-card">
              <AlertCircle size={32} />
              <h3>Analysis failed</h3>
              <p>{error}</p>
              <button className="secondary-button" onClick={() => runAnalysis()} disabled={!file || loading}>Try again</button>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div className={`result-icon ${isAI ? 'ai-result' : 'real-result'}`}>
                {isAI ? <ScanSearch size={29} /> : <CheckCircle2 size={29} />}
              </div>
              <div className="result-kicker">VERDICT</div>
              <h3 className={`result-label ${isAI ? 'text-ai' : 'text-real'}`}>{result.label}</h3>

              <div className="reason-banner">
                <Info size={15} />
                <span>{result.reason}</span>
              </div>

              <div className="confidence-heading">
                <span>Confidence score</span>
                <strong className={isAI ? 'text-ai' : 'text-real'}>{Number(result.confidence).toFixed(2)}%</strong>
              </div>
              <div className="confidence-track">
                <div
                  className={isAI ? 'track-ai' : 'track-real'}
                  style={{ width: `${Math.max(0, Math.min(100, Number(result.confidence)))}%` }}
                />
              </div>

              {/* Forensic Signals Breakdown */}
              {result.signals && (
                <div className="signals-grid">
                  <div className="signal-card">
                    <div className="signal-header">
                      <Camera size={14} />
                      <span>CAMERA HARDWARE</span>
                    </div>
                    <strong>
                      {result.signals.metadata?.has_camera_hardware
                        ? result.signals.metadata.camera_device || 'Verified Camera'
                        : 'No Hardware EXIF'}
                    </strong>
                    <span className="signal-sub">
                      {result.signals.metadata?.has_camera_hardware ? 'Authentic sensor signature' : 'Stripped or synthetic'}
                    </span>
                  </div>

                  <div className="signal-card">
                    <div className="signal-header">
                      <Sparkles size={14} />
                      <span>AI GENERATOR METADATA</span>
                    </div>
                    <strong>
                      {result.signals.metadata?.has_ai_metadata ? 'Fingerprint Found' : 'Clean / None'}
                    </strong>
                    <span className="signal-sub">
                      {result.signals.metadata?.has_ai_metadata ? 'Direct prompt/C2PA tags' : 'No generative headers'}
                    </span>
                  </div>

                  <div className="signal-card">
                    <div className="signal-header">
                      <Activity size={14} />
                      <span>2D FFT SPECTRUM</span>
                    </div>
                    <strong>
                      {result.signals.spectral?.spectral_anomaly ? 'Periodic Artifacts' : 'Natural Power-Law'}
                    </strong>
                    <span className="signal-sub">
                      Ratio: {result.signals.spectral?.high_freq_ratio}
                    </span>
                  </div>

                  <div className="signal-card">
                    <div className="signal-header">
                      <Cpu size={14} />
                      <span>NEURAL CLASSIFIER</span>
                    </div>
                    <strong>
                      AI {result.signals.neural_network?.ai_score}%
                    </strong>
                    <span className="signal-sub">
                      Auth: {result.signals.neural_network?.authentic_score}%
                    </span>
                  </div>
                </div>
              )}

              <div className="result-metrics">
                <div>
                  <span><Clock3 size={15} /> PROCESSING TIME</span>
                  <strong>{result.processing_time_ms} ms</strong>
                </div>
                <div>
                  <span><Layers size={15} /> ARCHITECTURE</span>
                  <strong>MobileNetV3-Large</strong>
                </div>
              </div>

              <button className="secondary-button full-button" onClick={reset}>
                <RotateCcw size={16} /> Analyze another image
              </button>
            </div>
          )}
        </div>
      </section>

      <footer className="footer">
        <span>VERITY <span className="brand-dot">/</span> IMAGE INTELLIGENCE</span>
        <span>RESEARCH PROTOTYPE · RESULTS ARE PROBABILISTIC</span>
      </footer>

      {cameraOpen && (
        <div className="modal-backdrop">
          <div className="camera-modal">
            <div className="camera-header">
              <h3>Capture an image</h3>
              <button className="icon-button" onClick={closeCamera}><X /></button>
            </div>
            <video ref={videoRef} autoPlay playsInline muted />
            <div className="camera-controls">
              <button className="secondary-button" onClick={closeCamera}>Cancel</button>
              <button className="analyze-button" onClick={capturePhoto}>
                <Camera size={18} /> Capture photo
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
