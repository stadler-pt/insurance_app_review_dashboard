FROM python:3.10-slim

# System-Tools installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces verlangen einen non-root User (uid 1000) für korrekte Schreibrechte
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Anforderungen kopieren und installieren (Besitzrechte an den User übergeben)
COPY --chown=user streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Gesamten Projektinhalt kopieren
COPY --chown=user . .

EXPOSE 7860

# Streamlit mit deaktiviertem CORS starten, damit der HF-Healthcheck nicht blockiert wird
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]