# Profilmining-Dienst

Der Dienst erzeugt aus hochgeladenen Dokumenten und URLs ein Profil und die Embeddings
für das Matchmaking der Conference-App. Entwurf und Umsetzung beschreiben Kapitel 5
und 6 der Arbeit.

Der Dienst lässt sich auf vier Arten betreiben:

|   | Betriebsart | Sprachmodell | Voraussetzung |
|---|---|---|---|
| A | Lokal, regelbasiert (Mock) | nein | Installation |
| B | Lokal mit Sprachmodell | ja | Installation, Zugangsschlüssel |
| C | Als Container | wahlweise | Docker |
| D | Im Verbund mit dem Backend | wahlweise | Docker, neighmash-Backend |

## Installation (für A und B)

Python 3.10 oder neuer, unter Linux und macOS oft als `python3` aufzurufen.

```
python -m venv venv
venv\Scripts\activate                         # Linux/macOS: source venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q                    # 96 Testfälle
```

`requirements.txt` enthält auch `torch` und `sentence-transformers` für Betriebsart B.
Unter Linux holt pip dabei die CUDA-Bibliotheken von torch mit, zusammen rund 5,5 GB.
Wer keine Grafikkarte nutzt, installiert vorher die CPU-Variante:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## A · Lokal ohne Sprachmodell (Mock)

```
python -m uvicorn service:app --port 8080     # Oberfläche: http://localhost:8080/docs
```

Der Dienst arbeitet mit dem regelbasierten Provider, ein Schlüssel ist nicht nötig.
Jeder Aufruf braucht den Header `X-User-Id` mit einem beliebigen Wert, auch in der
Oberfläche, sonst antwortet der Dienst mit 401.

---

## B · Lokal mit Sprachmodell

Anbieter, Modell und Schlüssel stehen in der Vorlage `llm.env` in der obersten Ebene
der Abgabe (von hier aus `../../llm.env`); dort nur noch den Schlüssel eintragen.
Voreingestellt ist die Uni-LLM-API:

```
PROFILE_MINING_PROVIDER=uni
LLM_BASE_URL=https://chat-ai.academiccloud.de/v1
LLM_MODEL=gemma-4-31b-it
UNI_LLM_API_KEY=
```

Für Mistral, OpenAI oder Anthropic stattdessen `PROFILE_MINING_PROVIDER=mistral`
(bzw. `openai`, `anthropic`) und `MISTRAL_API_KEY` (bzw. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`).

Variablen laden und starten:

```
# Windows (PowerShell)
Get-Content ..\..\llm.env | Where-Object { $_ -match '^\w+=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
python -m uvicorn service:app --port 8080

# Linux/macOS
set -a; . ../../llm.env; set +a
python3 -m uvicorn service:app --port 8080
```

Beim ersten Job lädt der Dienst einmalig das Embedding-Modell von Hugging Face
(rund 470 MB) und hält es danach im Speicher. `/healthz` zeigt den aktiven Anbieter.
Alle Variablen stehen in Anhang D.3.

---

## C · Als Container

```
docker compose up --build                                # Mock
docker compose --env-file ../../llm.env up --build       # mit Sprachmodell, llm.env wie bei B
```

Der Dienst ist dann wie bei A unter `http://localhost:8080` erreichbar, ebenfalls mit
dem Header `X-User-Id`. Ist der Port belegt, vorher die Umgebungsvariable `HOST_PORT`
setzen, etwa auf `8081`.

---

## D · Im Verbund mit dem Backend

Der Dienst läuft als Container hinter Traefik und Keycloak des neighmash-Backends der
Conference-App. Das Backend liegt der Abgabe nicht bei.

1. `traefik_profilemining.yaml` als `traefik/dynamic/profilemining.yaml` in den Stack
   des Backends kopieren. Die Route lässt jedes Token vom Backend prüfen, bevor die
   Anfrage den Dienst erreicht.
2. Backend-Stack starten, dann den Dienst dazuschalten:

   ```
   docker network create app-public      # nur, wenn das Netz noch nicht existiert
   docker compose -f compose.yaml -f compose.verbund.yaml --env-file ../../llm.env up -d --build
   ```

   Ohne `--env-file` läuft der Dienst auch im Verbund mit dem regelbasierten Provider.
3. Der Dienst ist jetzt nur noch über `https://docker.localhost/api/...` erreichbar
   und braucht ein Keycloak-Token. Ob die Token-Prüfung greift, zeigt:

   ```
   python tests/verbund_auth_check.py --base https://docker.localhost/api --token "<Token>"
   ```

`API_PREFIX` bleibt im Verbund leer, weil Traefik das Präfix `/api` entfernt.

---

## Dateien

| Datei | Inhalt |
|---|---|
| `service.py` | HTTP-Schnittstelle, Jobverwaltung, Zugriffsschutz |
| `pipeline.py` | Mining-Pipeline |
| `provider.py` | Anbindung der Sprachmodelle und regelbasierter Provider |
| `embedding_compression.py` | Kürzen und Quantisieren der Vektoren |
| `cache.py` | Ergebnisspeicher mit einmaligem Abruf |
| `openapi.yaml` | Schnittstellenbeschreibung, erzeugt mit `export_openapi.py` |
| `tests/` | Unit-Tests und zusätzliche Prüfskripte |
| `run_demo.py`, `sample_docs/` | Minet ein Profil aus Beispieldokumenten |
| `demo_kolloquium.py` | Spielt den Ablauf gegen einen laufenden Dienst durch |
| `compose.verbund.yaml`, `traefik_profilemining.yaml` | Betrieb im Verbund |
