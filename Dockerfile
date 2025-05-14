FROM python:3.11-slim

LABEL maintainer="Morningstar Team <contact@morningstar.ai>"
LABEL version="2.0.0"
LABEL description="Image Docker pour le modèle monolithique Morningstar"

# Arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_DEFAULT_TIMEOUT=100

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt trading_env_deps.txt ./

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r trading_env_deps.txt

# Copier le code source
COPY ultimate /app/ultimate
COPY config /app/config
COPY scripts /app/scripts

# Créer les répertoires nécessaires
RUN mkdir -p /app/data/raw /app/data/processed /app/logs /app/ultimate/monitoring/metrics

# Exposer les ports (API REST, Prometheus)
EXPOSE 8000
EXPOSE 8888

# Volume pour les données persistantes
VOLUME ["/app/data", "/app/logs", "/app/ultimate/monitoring/metrics"]

# Utilisateur non-root pour la sécurité
RUN groupadd -g 1000 morningstar && \
    useradd -u 1000 -g morningstar -s /bin/bash -m morningstar && \
    chown -R morningstar:morningstar /app

# Changer vers l'utilisateur non-root
USER morningstar

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée et commande par défaut
ENTRYPOINT ["python", "-m"]
CMD ["ultimate.scripts.run_live", "--testnet"] 