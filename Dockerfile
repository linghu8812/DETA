# docker build -t DETA:2023.8.21 .
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

ADD . /workspace/DETA
WORKDIR /workspace/DETA
#RUN echo 'Etc/UTC' > /etc/timezone &&  rm -f /etc/localtime &&  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#    apt-get update && apt-get install tzdata -y

RUN bash -xc "apt-get update && apt-get install ssh vim git zip tmux p7zip-full python3-pip -y \
    && ln -s /usr/bin/python3 /usr/bin/python && pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install pycocotools tqdm cython scipy tensorboard onnx onnx-simplifier onnxruntime timm -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pushd models/ops 2>&1 > /dev/null && sh make.sh && popd 2>&1 > /dev/null"
