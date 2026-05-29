FROM python:3.10-slim

# System-Tools für Pakete installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. Anforderungen aus dem Unterordner kopieren und installieren
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Den gesamten Projektinhalt (inkl. output/ mit den Modellen) kopieren
COPY . .

# Hugging Face Spaces Port freigeben
EXPOSE 7860

# 3. Streamlit direkt aus dem Unterordner starten
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]