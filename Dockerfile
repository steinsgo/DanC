FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app /saisresult

WORKDIR /app

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    "numpy<2" ultralytics opencv-python-headless Pillow

COPY models/ /app/models/
COPY scripts/run_inference.py /app/src/run_inference.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["bash", "/app/run.sh"]
