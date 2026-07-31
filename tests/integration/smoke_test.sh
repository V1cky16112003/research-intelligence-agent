#!/usr/bin/env bash
# Smoke test against deployed backend
# Run from GitHub Actions or locally with SERVICE_URL set

set -euo pipefail

SERVICE_URL="${SERVICE_URL:-https://$(aws apprunner describe-service --service-arn ${APP_RUNNER_SERVICE_ARN} --query 'Service.ServiceUrl' --output text)}"
echo "Testing: $SERVICE_URL"

# Health check
echo "→ Health check..."
curl -sf "$SERVICE_URL/health" | jq .

# Chat endpoint
echo "→ Chat test..."
RESPONSE=$(curl -sf -X POST "$SERVICE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "smoke-test"}')
echo "$RESPONSE" | jq .

# Verify response structure
echo "$RESPONSE" | jq -e '.answer and .citations and .session_id' > /dev/null
echo "✅ Response structure valid"

# Check provider used
PROVIDER=$(echo "$RESPONSE" | jq -r '.provider')
echo "Provider: $PROVIDER"

echo "=== Smoke test passed ==="