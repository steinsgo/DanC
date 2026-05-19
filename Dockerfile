FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app /saisresult && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tini bash python3.11 python3-pip \
        libgl1-mesa-glx libglib2.0-0 libgomp1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY scripts/run_inference.py /app/src/run_inference.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/sbin/tini", "--", "bash", "/app/run.sh"]
