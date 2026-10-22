FROM python:3.11-slim
WORKDIR /workspace
COPY services/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY services /workspace/services
COPY packages /workspace/packages
RUN pip install --no-cache-dir /workspace/packages/fastreid
EXPOSE 8000
CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
