#FROM python:3.10-buster
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1
ENV JAVA_HOME=/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server/
ENV EMTSV_NUM_PROCESSES=2
ENV host=host.docker.internal

RUN rm -rf /etc/apt/sources.list.d/cuda.list
RUN rm -rf /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    parallel
RUN mv /bin/sh /bin/sh.orig && ln -s /bin/bash /bin/sh

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3
WORKDIR /tmp
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt install -y python3.10-distutils && \
    python3.10 get-pip.py
RUN apt-get remove -y wget vim git cmake automake ant curl && apt-get clean

ARG http_proxy=""
COPY ./requirements.txt /tmp/requirements.txt
RUN rm -rf /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --no-cache-dir --proxy=${http_proxy} --upgrade pip && \
    python -m pip install --no-cache-dir --proxy=${http_proxy} cython numpy
RUN python -m pip install --no-cache-dir --proxy=${http_proxy} -r /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --proxy=${http_proxy} https://huggingface.co/huspacy/hu_core_news_trf/resolve/main/hu_core_news_trf-any-py3-none-any.whl

RUN apt install -y git curl git-lfs

WORKDIR /app

COPY ./install.sh /app/install.sh

COPY ./contents /app/contents
COPY ./entrypoint.sh /app/entrypoint.sh
COPY ./src /app/src
