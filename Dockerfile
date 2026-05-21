FROM python:3.11-slim

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app /saisresult && \
    apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    "numpy<2" && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    torch==2.2.0 torchvision==0.17.0 && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    ultralytics opencv-python-headless Pillow

COPY models/ /app/models/
COPY scripts/run_inference.py /app/src/run_inference.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["bash", "/app/run.sh"]
