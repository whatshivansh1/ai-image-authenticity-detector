# ==============================================================================
# AWS Lambda Container with AWS Lambda Web Adapter
# ==============================================================================
# Allows standard FastAPI/Uvicorn applications to run on AWS Lambda Serverless
# with zero code modifications, sub-50ms warm execution, and 1M free requests/mo.
# ==============================================================================

FROM public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 AS aws-adapter
FROM python:3.11-slim

# Copy the AWS Lambda Web Adapter extension binary
COPY --from=aws-adapter /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

# Install lightweight dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights and backend code
COPY models/ models/
COPY backend/ backend/

ENV PORT=8000
EXPOSE 8000

# Lambda Web Adapter proxies incoming events to localhost:8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
