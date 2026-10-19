FROM python:3.11-slim
WORKDIR /workspace
COPY services/worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY services /workspace/services
COPY packages /workspace/packages
CMD ["python", "-m", "services.worker.app.worker"]
