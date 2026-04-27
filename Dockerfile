FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt requirements-app.txt
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["bash", "scripts/05_label_ui.sh"]
