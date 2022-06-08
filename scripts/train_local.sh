#!/bin/bash

echo ${PWD}

echo "Submitting AI Platform PyTorch job"

# BUCKET_NAME: Change to your bucket name.

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=toy_ml_container

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=image1

# The PyTorch image provided by AI Platform Training.
#IMAGE_URI=container/${IMAGE_REPO_NAME}_${IMAGE_TAG}
IMAGE_URI=${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
#JOB_PREFIX=toy_ml_proj
#JOB_NAME=${JOB_PREFIX}_$(date +%Y%m%d_%H%M%S)

# Build the docker image
#docker build -f Dockerfile -t ${IMAGE_URI} ../
#docker build -t ${IMAGE_URI} ../ # old command, creates a docker image successfully

docker buildx build --platform=linux/arm64/v8 -t ${IMAGE_URI} ../

#docker build -f Dockerfile -t  .
#docker build -t ${IMAGE_URI} .

# Deploy the docker image to Cloud Container Registry
#docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

