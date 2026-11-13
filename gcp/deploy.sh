#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Run 1-Click Serverless Deployment (Always Free Tier)
# ==============================================================================
# Free Tier Allowances:
#   - 2,000,000 requests / month
#   - 360,000 GB-seconds memory / month
#   - 180,000 vCPU-seconds / month
#   - In-flight execution latency: ~30-40 ms on 1 vCPU
# ==============================================================================

set -euo pipefail

SERVICE_NAME="${1:-ai-image-detector-backend}"
REGION="${2:-us-central1}"

echo "[*] Deploying ${SERVICE_NAME} to Google Cloud Run in ${REGION}..."

# Enable required Google Cloud APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudscheduler.googleapis.com

# Deploy container directly from source directory
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 80 \
  --timeout 30s

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format="value(status.url)")

echo ""
echo "[✓] Deployment Complete!"
echo "Service URL: ${SERVICE_URL}"
echo "Health Check: ${SERVICE_URL}/health"
echo ""
echo "Update your Vercel VITE_API_URL environment variable to: ${SERVICE_URL}"
