#!/bin/bash
set -e

AWS_REGION="ap-south-1"
AWS_ACCOUNT_ID="165098158976"
ECR_REPOSITORY="rounak_mlflow"
IMAGE_TAG="v3"
IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

# Login to AWS ECR
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Pull the latest image
docker pull "$IMAGE"

# Check if the container 'sentiment-app' is running
if [ "$(docker ps -q -f name=sentiment-app)" ]; then
    docker stop sentiment-app
fi

# Check if the container 'sentiment-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=sentiment-app)" ]; then
    docker rm sentiment-app
fi

# Run a new container — DAGSHUB_PAT read from a file on disk, never
# hardcoded in this script (see setup step: /home/ubuntu/.dagshub_pat)
docker run -d \
  -p 5001:5001 \
  -e DAGSHUB_PAT=47bf3b542aec24b9bd183e1332212cfd2c8c3d19\
  --restart unless-stopped \
  --name sentiment-app \
  "$IMAGE"