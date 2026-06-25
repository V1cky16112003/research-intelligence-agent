FROM python:3.11-slim
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/.cache/sentence_transformers
ENV MLFLOW_TRACKING_URI=""
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
