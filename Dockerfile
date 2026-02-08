FROM python:3.10-slim

# ===============================
# Basic env safety
# ===============================
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ===============================
# System dependencies
# ===============================
# - poppler-utils → pdf2image
# - tesseract-ocr → EasyOCR backend
# - libgl1 + libglib → OpenCV / EasyOCR runtime
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Workdir
# ===============================
WORKDIR /app

# ===============================
# Python deps (cached layer)
# ===============================
COPY requirements.txt .

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu


RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ===============================
# App code
# ===============================
COPY . .

# ===============================
# Render port (DO NOT change)
# ===============================
EXPOSE 10000

# ===============================
# Start FastAPI (Render-compatible)
# ===============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
