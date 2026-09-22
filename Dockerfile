FROM python:3.13-slim
LABEL qfbench2.interface_version="2.0"

ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    HF_HOME=/tmp \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY strong_rag_baseline /app/strong_rag_baseline
COPY agent.py /app/agent.py

ENTRYPOINT ["python", "/app/agent.py"]
CMD ["analyze", "--help"]
