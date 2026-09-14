FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements*.txt ./
# The image serves the HTTP API, so it needs the API bundle — installing only
# requirements.txt produced an image whose CMD (uvicorn) was not installed.
RUN pip install --user --no-cache-dir -r requirements-api.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
