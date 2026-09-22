FROM python:3.13-slim

LABEL qfbench2.interface_version="2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    T4_MODEL_RETRIES=1 \
    T4_TOP_K=12 \
    T4_MODEL_TIMEOUT_S=90

WORKDIR /app
COPY strong_rag_baseline ./strong_rag_baseline
COPY auction_family.py ./auction_family.py
COPY main.py ./main.py

ENTRYPOINT ["python", "/app/main.py"]
CMD ["analyze", "--help"]
