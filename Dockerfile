FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# XGBoost requires the OpenMP runtime on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 1000 student
COPY --chown=student:student src/ ./src/
COPY --chown=student:student notebooks/ ./notebooks/
COPY --chown=student:student data/ ./data/
COPY --chown=student:student models/ ./models/
COPY --chown=student:student reports/ ./reports/
COPY --chown=student:student README.md ./
RUN chown student:student /app

USER student
WORKDIR /app/notebooks
EXPOSE 8888

CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app"]
