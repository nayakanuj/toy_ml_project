# Install pytorch
#FROM gcr.io/cloud-ml-public/training/pytorch-gpu.1-9
#FROM pytorch/pytorch
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
#FROM node

RUN mkdir container
RUN touch container/__init__.py

WORKDIR container

# Installs pandas, and google-cloud-storage.
#RUN pip install google-cloud-storage torch torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric scikit-learn numpy
#RUN pip install torch torchvision numpy

# Copies the trainer code to the docker image.
COPY search_hyperparams.py  .
COPY synthesize_results.py  .
COPY main.py		        .
ADD experiments	            experiments
ADD model		            model
COPY utils.py		        .
COPY train.py		        .
COPY evaluate.py	        .
ADD data                    data

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "train"]
