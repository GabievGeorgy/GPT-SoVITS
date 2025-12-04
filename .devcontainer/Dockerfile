ARG VARIANT=3.10
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

ARG TORCH_CHANNEL=cpu
ARG TORCH_VERSION=2.4.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git-lfs \
        libopencc-dev \
        libsndfile1 \
        libmecab-dev \
        mecab \
        mecab-ipadic-utf8 \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/devcontainer

COPY requirements.txt extra-req.txt ./

RUN python -m pip install --no-cache-dir --upgrade pip

RUN if [ "${TORCH_CHANNEL}" = "cu128" ]; then \
        python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}; \
    elif [ "${TORCH_CHANNEL}" = "cu126" ]; then \
        python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}; \
    else \
        python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}; \
    fi

RUN python -m pip install --no-cache-dir -r extra-req.txt --no-deps \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && (python -m pip uninstall -y onnxruntime-gpu >/dev/null 2>&1 || true) \
    && python -m pip install --no-cache-dir onnxruntime \
    && python -m pip install --no-cache-dir pytest \
    && python -m pip cache purge

ENV PYTHONPATH=/workspaces/GPT-SoVITS

WORKDIR /workspaces/GPT-SoVITS

USER vscode
