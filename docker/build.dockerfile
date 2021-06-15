ARG CUDA_VERSION=11.1.1
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

LABEL maintainer="Dong Wang"


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y wget

# Python
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN pip3 install --upgrade pip

WORKDIR /workspace
ADD ./requirements.txt /workspace
RUN pip3 install -r /workspace/requirements.txt and &&\
	rm /workspace/requirements.txt

CMD ["/bin/bash"]