FROM runpod/base:0.6.3-cuda12.1.0

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py
COPY schemas.py /workspace/schemas.py
COPY download_weights.py /workspace/download_weights.py
COPY test_input.json /workspace/test_input.json

ENV HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
