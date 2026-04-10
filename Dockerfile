# ╔══════════════════════════════════════════════════════════╗
# ║        WAHOU — Dockerfile                                ║
# ║        Guichet Unique du Développement des PME           ║
# ╚══════════════════════════════════════════════════════════╝

# ── Base image ──────────────────────────────────────────────
FROM python:3.12-slim

# ── Métadonnées ─────────────────────────────────────────────
LABEL maintainer="GUDE — Côte d'Ivoire"
LABEL description="Wahou — Assistante Virtuelle du GUDE"
LABEL version="1.0.0"

# ── Variables d'environnement ────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ── Répertoire de travail ────────────────────────────────────
WORKDIR /app

# ── Dépendances système (nécessaires pour HuggingFace / torch) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ───────────────────────────────────────
# Copier d'abord requirements.txt seul pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copie du code source ─────────────────────────────────────
COPY . .

# ── Vérification que les dossiers critiques sont présents ────
RUN mkdir -p static templates vector_bd

# ── Port exposé ──────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Lancement ────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]