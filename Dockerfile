FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/petrosite

COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY app.py .
COPY .streamlit .streamlit
COPY static static

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv/petrosite
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read()==b'ok' else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
