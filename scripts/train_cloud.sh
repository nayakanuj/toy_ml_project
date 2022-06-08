#!/bin/bash

echo "Submitting AI Platform PyTorch job"

mkdir container
cp -R ../search_hyperparams.py    ./container/search_hyperparams.py
cp -R ../synthesize_results.py    ./container/synthesize_results.py
cp -R ../main.py		              ./container/main.py
cp -R ../experiments	            ./container/experiments
cp -R ../model		                ./container/model
cp -R ../utils.py		              ./container/utils.py
cp -R ../train.py		              ./container/train.py
cp -R ../evaluate.py	            ./container/evaluate.py

touch container/__init__.py

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME=test_bucket_neuro2

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=toy_ml_container

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=toy_ml_image

# The PyTorch image provided by AI Platform Training.
#IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
JOB_PREFIX=toy_ml_proj
JOB_NAME=${JOB_PREFIX}_$(date +%Y%m%d_%H%M%S)

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

# REGION: select a region from https://cloud.google.com/ai-platform/training/docs/regions
REGION=us-central1

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_PREFIX}/gcloud_models/${JOB_NAME}
PACKAGE_PATH=container

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC \
    --job-dir ${JOB_DIR} \
    --module-name train.py \
    --package-path ${PACKAGE_PATH} \
    -- \
    --model-name="two_layer_net"

#--scale-tier=CUSTOM \
#--master-machine-type=n1-standard-8 \
#--master-accelerator=type=nvidia-tesla-t4,count=2 \
#nvidia-tesla-t4,count=2 \

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
sudo gsutil ls ${JOB_DIR}/
