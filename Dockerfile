# Group 08 Streamlit app - container image
# Build:  docker build -t group08-streamlit .
# Run:    docker run --rm -p 8501:8501 group08-streamlit      ->  http://localhost:8501
FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# dependencies first, so code changes do not invalidate this (slow) layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY students/ students/
COPY .streamlit/config.toml .streamlit/config.toml

# run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# API URLs can be overridden at run time, e.g. -e STUDENT_A_API_URL=https://...
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
