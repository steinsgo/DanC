FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app /saisresult && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tini bash libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir ultralytics>=8.1.0 opencv-python-headless>=4.8.0 Pillow>=10.0.0

COPY models/ /app/models/
COPY scripts/run_inference.py /app/src/run_inference.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/app/run.sh"]
