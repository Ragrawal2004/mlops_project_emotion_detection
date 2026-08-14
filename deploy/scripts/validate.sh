#!/bin/bash
# Confirms the newly-started container is actually serving requests —
# not just that `docker run` succeeded. If this script exits non-zero,
# CodeDeploy marks the whole deployment as FAILED.

for i in {1..15}; do
  if curl -fs http://localhost:80/health > /dev/null; then
    echo "Service is healthy."
    exit 0
  fi
  echo "Waiting for service to become healthy... ($i/15)"
  sleep 3
done

echo "Service failed health check!"
docker logs sentiment-app
exit 1