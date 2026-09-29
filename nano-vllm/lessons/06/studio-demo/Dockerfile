FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG MINI_SGLANG_COMMIT=9a91cfafe754aa85daee49998176275667eb58f2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/mnt/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

RUN git clone https://github.com/sgl-project/mini-sglang.git /opt/mini-sglang \
    && git -C /opt/mini-sglang checkout "${MINI_SGLANG_COMMIT}"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -e /opt/mini-sglang \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
