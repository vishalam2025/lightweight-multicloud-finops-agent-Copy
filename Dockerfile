FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py detectors.py evaluate.py prediction.py ./
COPY evaluate_hybrid_final.py evaluate_azure.py evaluate_alibaba.py ./
COPY benchmark_workload.py dashboard.py ./

EXPOSE 8501

CMD ["python", "main.py"]