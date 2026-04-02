FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

WORKDIR /workspace/ApDepth

COPY environment.yaml requirements.txt requirements+.txt requirements++.txt ./

RUN conda create -n apdepth python==3.12.9 -y

RUN /opt/conda/envs/apdepth/bin/pip install --no-cache-dir -r requirements.txt
RUN /opt/conda/envs/apdepth/bin/pip install --no-cache-dir -r requirements+.txt\
RUN /opt/conda/envs/apdepth/bin/pip install --no-cache-dir -r requirements++.txt

COPY . .

SHELL ["conda", "run", "-n", "apdepth", "/bin/bash", "-c"]

CMD ["/bin/bash"]