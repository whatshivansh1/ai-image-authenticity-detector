#!/usr/bin/env bash
# ==============================================================================
# AWS Lambda Serverless 1-Click Deployment (Always Free Tier)
# ==============================================================================
# Free Tier Allowances:
#   - 1,000,000 free requests / month
#   - 3,200,000 seconds of compute time / month
#   - In-flight execution latency: ~35-50 ms with 2048 MB RAM
# ==============================================================================

set -euo pipefail

STACK_NAME="${1:-ai-image-detector-stack}"
REGION="${2:-us-east-1}"

echo "[*] Building and deploying ${STACK_NAME} to AWS Lambda via SAM (${REGION})..."

# Check if AWS CLI and SAM CLI are installed
if ! command -v sam &> /dev/null; then
    echo "[!] AWS SAM CLI not found. Please install sam-cli or use manual ECR container deployment."
    echo "[*] Guide: brew install aws-sam-cli"
    exit 1
fi

sam build --template aws/template.yaml

sam deploy \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --resolve-s3 \
  --resolve-image-repos \
  --capabilities CAPABILITY_IAM \
  --no-confirm-changeset

echo ""
echo "[✓] AWS Lambda Deployment Complete!"
sam list stack-outputs --stack-name "${STACK_NAME}" --region "${REGION}"
