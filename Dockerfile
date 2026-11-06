# Profilmining-Dienst als eigenstaendiger Container (vgl. Ausarbeitung, Abschnitt Deployment)
# Build:  docker build -t profile-mining .
# Start:  docker run --rm -p 8080:8080 profile-mining
#         Default: Mock-Provider (kein API-Key noetig); Konfiguration per -e,
#         z. B. -e PROFILE_MINING_PROVIDER=uni -e LLM_BASE_URL=... -e UNI_LLM_API_KEY=...
FROM python:3.12-slim

WORKDIR /app

# Erst die Abhaengigkeiten (Layer-Cache), dann der Code.
# torch vorab als CPU-Variante: sonst zieht requirements.txt unter Linux die
# CUDA-Variante mit mehreren GB Bibliotheken, die im Container niemand braucht.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Kein lokales Modell im Container: LLM laeuft beim Managed-Anbieter (Architekturentscheidung)
ENV PROFILE_MINING_PROVIDER=mock \
    API_PREFIX=""

# Dienst laeuft unter einem unprivilegierten Konto. Vorher lief der Prozess als
# root: eine Luecke im Dienst haette damit unmittelbar root-Rechte im Container
# bedeutet. Das Cache-Verzeichnis muss dem Konto gehoeren, sonst kann
# sentence-transformers das Embedding-Modell nicht ablegen (vgl. compose.yaml).
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app /app
USER app
ENV HF_HOME=/home/app/.cache/huggingface

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
