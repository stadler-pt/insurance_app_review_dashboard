FROM python:3.10-slim

# Install required system tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces often expects a non-root user with UID 1000
# so the app can write to mounted directories with the correct permissions.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/data/.huggingface \
    HF_HUB_CACHE=/data/.huggingface/hub \
    TRANSFORMERS_CACHE=/data/.huggingface/transformers \
    HF_DATASETS_CACHE=/data/.huggingface/datasets
WORKDIR $HOME/app

# Copy and install Python dependencies first to improve Docker layer caching.
# The files are copied with user ownership so pip can work without root.
COPY --chown=user streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container.
COPY --chown=user . .

EXPOSE 7860

# Start Streamlit with CORS and XSRF protection disabled so that Hugging Face Spaces health checks and embedded execution do not get blocked.
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none", "--server.headless=true"]