FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update 
RUN apt-get install -y git python3 python3-pip
WORKDIR /workspace

RUN git clone https://github.com/prs-eth/Marigold.git
RUN cd /workspace/Marigold && pip install -r requirements.txt

WORKDIR /workspace/Marigold