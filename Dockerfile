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
    T4_MODEL_TIMEOUT_S=90 \
    T4_MODEL_MAX_TOKENS=3000 \
    T4_UNIT_TIMEOUT_S=540

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY strong_rag_baseline ./strong_rag_baseline
COPY auction_family.py ./auction_family.py
COPY eps_growth_family.py ./eps_growth_family.py
COPY safe_calibration.py ./safe_calibration.py
COPY ARTIFACT_PROVENANCE.md ./ARTIFACT_PROVENANCE.md
COPY output_contract.py ./output_contract.py
COPY analysis.schema.json ./analysis.schema.json
COPY main.py ./main.py

ENTRYPOINT ["python", "/app/main.py"]
CMD ["analyze", "--help"]
