#!/usr/bin/env python3
"""Empirische Gegenpruefung der Audit-Befunde gegen den laufenden Dienst."""
import os, sys, json, threading, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ["PROFILE_MINING_PROVIDER"] = "mock"
from fastapi.testclient import TestClient
import service

c = TestClient(service.app)
R = []


def check(nr, name, ok, detail=""):
    R.append((nr, name, ok, detail))
    print(f"[{'BESTAETIGT' if ok else 'nicht reproduziert'}] {nr} {name}: {detail}")


# --- 1: Identitaetsuebernahme per X-User-Id -----------------------------------
c.post("/profiling", headers={"X-User-Id": "opfer-alice"})
c.post("/profiling/documents", headers={"X-User-Id": "opfer-alice"},
       files={"file": ("cv.txt", b"Ich bin Doktorandin und arbeite an Deep Learning "
                                 b"und Reinforcement Learning. Suche Kooperation.", "text/plain")})
c.patch("/profiling", headers={"X-User-Id": "opfer-alice"}, json={"status": "submitted"})
for _ in range(50):
    if c.get("/profiling", headers={"X-User-Id": "opfer-alice"}).json()["status"] == "succeeded":
        break
    time.sleep(0.05)
# Angreifer setzt nur den Header:
r = c.get("/profiling/result", headers={"X-User-Id": "opfer-alice"})
check(1, "Fremdes Ergebnis per X-User-Id abrufbar", r.status_code == 200,
      f"HTTP {r.status_code}, Felder={list(r.json().get('profile', {}).keys())[:3]}")

# --- 2: JWT ohne Signaturpruefung --------------------------------------------
import base64
def unsigned_jwt(sub):
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{b({'alg':'none'})}.{b({'sub':sub})}.GEFAELSCHTE-SIGNATUR"
c.post("/profiling", headers={"Authorization": "Bearer " + unsigned_jwt("opfer-bob")})
r = c.get("/profiling", headers={"X-User-Id": "opfer-bob"})
check(2, "Selbstgebautes JWT ohne gueltige Signatur akzeptiert", r.status_code == 200,
      f"Job von 'opfer-bob' existiert: HTTP {r.status_code}")

# --- 3: Anonyme Anfragen teilen EINE Identitaet ('demo-user') ----------------
r1 = c.post("/profiling")            # ohne jeden Header
r2 = c.post("/profiling")            # zweiter, voellig anderer Client
check(3, "Anfrage ohne Auth wird zu Sammelidentitaet 'demo-user'",
      r1.status_code == 201 and r2.status_code == 409,
      f"1. Client HTTP {r1.status_code}, 2. Client HTTP {r2.status_code} (409 = kollidiert)")

# --- 4: URL-Liste unbegrenzt -------------------------------------------------
c.post("/profiling", headers={"X-User-Id": "dos1"})
N = 20000
r = c.post("/profiling/urls", headers={"X-User-Id": "dos1"},
           json={"urls": [{"url": f"https://example.com/{i}"} for i in range(N)]})
n = len(c.get("/profiling/urls", headers={"X-User-Id": "dos1"}).json())
grenze = getattr(service, "MAX_URLS_PER_JOB", None)
check(4, "URL-Zahl je Job begrenzt", grenze is not None and n <= grenze,
      f"{N} URLs angeboten, HTTP {r.status_code}, {n} gespeichert; "
      f"MAX_URLS_PER_JOB={grenze}, Fetch-Budget="
      f"{getattr(service, 'URL_FETCH_BUDGET_S', None)}s")

# --- 5: Doppeltes PATCH startet zwei Worker (fehlende Sperre) ----------------
starts = []
orig = service.run_job
def spy(user):
    starts.append(user)
    time.sleep(0.2)
    return orig(user)
service.run_job = spy
c.post("/profiling", headers={"X-User-Id": "race1"})
c.post("/profiling/documents", headers={"X-User-Id": "race1"},
       files={"file": ("a.txt", b"Machine Learning und Computer Vision, Doktorand, suche Kooperation.", "text/plain")})
def submit():
    c.patch("/profiling", headers={"X-User-Id": "race1"}, json={"status": "submitted"})
ts = [threading.Thread(target=submit) for _ in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
service.run_job = orig
check(5, "Gleichzeitiges PATCH startet mehrere Worker fuer denselben Job",
      len(starts) > 1, f"{len(starts)} Worker-Starts bei 8 parallelen PATCH")

# --- 6: Upload-Grenze nicht atomar -> mehr als MAX_DOCS_PER_JOB -------------
c.post("/profiling", headers={"X-User-Id": "race2"})
def up(i):
    c.post("/profiling/documents", headers={"X-User-Id": "race2"},
           files={"file": (f"d{i}.txt", b"x" * 100, "text/plain")})
ts = [threading.Thread(target=up, args=(i,)) for i in range(60)]
[t.start() for t in ts]; [t.join() for t in ts]
n = len(c.get("/profiling/documents", headers={"X-User-Id": "race2"}).json())
check(6, f"Dokumentgrenze ({service.MAX_DOCS_PER_JOB}) unter Parallelitaet ueberschreitbar",
      n > service.MAX_DOCS_PER_JOB, f"{n} Dokumente im Job")

# --- 7: Defektes PDF -> unbehandelte Ausnahme (HTTP 500) --------------------
c.post("/profiling", headers={"X-User-Id": "pdf1"})
try:
    r = c.post("/profiling/documents", headers={"X-User-Id": "pdf1"},
               files={"file": ("boese.pdf", b"%PDF-1.4\nkein gueltiges PDF" + b"\x00" * 50,
                               "application/pdf")})
    code, exc = r.status_code, ""
except Exception as e:
    code, exc = 500, type(e).__name__
check(7, "Defektes PDF fuehrt zu HTTP 500 statt 400/422", code >= 500,
      f"HTTP {code} {exc} (nur ImportError wird gefangen)")

# --- 8: Interne Fehlermeldung landet im Statusfeld --------------------------
c.post("/profiling", headers={"X-User-Id": "leak1"})
c.post("/profiling/documents", headers={"X-User-Id": "leak1"},
       files={"file": ("a.txt", b"Deep Learning Doktorand mit Fokus auf Transformer und Attention.", "text/plain")})
import provider as prov
_orig_extract = prov.MockProvider.extract_signals
def boom(self, chunk, lang):
    raise RuntimeError("uni-API: HTTP 401. Antwort: {'error':'invalid api key sk-abc123...'}")
prov.MockProvider.extract_signals = boom
c.patch("/profiling", headers={"X-User-Id": "leak1"}, json={"status": "submitted"})
for _ in range(50):
    j = c.get("/profiling", headers={"X-User-Id": "leak1"}).json()
    if j["status"] == "failed":
        break
    time.sleep(0.05)
prov.MockProvider.extract_signals = _orig_extract
stage = (j.get("stage") or "")
check(8, "Roher Ausnahmetext wird an den Client ausgeliefert",
      "invalid api key" in stage, f"stage={stage[:90]!r}")

# --- 9: Wird der Provider (und damit das Embedding-Modell) wiederverwendet? --
# Gezaehlt werden verschiedene OBJEKTE, nicht Aufrufe: nur eine geteilte Instanz
# vermeidet, dass jede Anfrage ihr eigenes Embedding-Modell laedt.
ids = set()
_mk = service.make_provider
def counting():
    p = _mk(); ids.add(id(p)); return p
service.make_provider = counting
for i in range(5):
    c.post("/embeddings", headers={"X-User-Id": "audit"},
           json={"description": "Deep Learning und Computer Vision, Doktorand."})
service.make_provider = _mk
check(9, "Provider wird prozessweit wiederverwendet (ein Modell-Ladevorgang)",
      len(ids) == 1, f"{len(ids)} verschiedene Instanzen bei 5 Anfragen "
                     f"(1 = geteilt, 5 = je Anfrage ein Modell-Ladevorgang)")

# --- 10: Speicherbedarf je Nutzer -------------------------------------------
per_user = service.MAX_DOCS_PER_JOB * service.MAX_DOC_BYTES
gesamt = getattr(service, "MAX_JOBS_TOTAL", None)
check(10, "Speicherbedarf je Nutzer unbegrenzt nach oben", gesamt is None,
      f"{service.MAX_DOCS_PER_JOB} Dok. x {service.MAX_DOC_BYTES/1e6:.0f} MB "
      f"= {per_user/1e6:.0f} MB/Nutzer; "
      + ("keine Obergrenze fuer die Zahl gehaltener Jobs"
         if gesamt is None else
         f"Jobs insgesamt auf {gesamt} begrenzt -> Worst Case {per_user*gesamt/1e9:.0f} GB, "
         f"in der Praxis weit darunter, weil die wenigsten Entwuerfe ausgereizt werden"))

# --- 11: sweep_jobs O(n) unter globaler Sperre bei jedem Polling ------------
# Gemessen wird die ABHAENGIGKEIT von der Jobzahl, nicht die absolute Zeit: der
# Grundaufwand einer Anfrage (Routing, Validierung, Serialisierung) liegt bei
# rund 2 ms und wuerde eine absolute Schwelle unabhaengig vom Befund reissen.
def polling_dauer(n):
    service.JOBS.clear()
    service._sweep_due = float("inf")
    c.post("/profiling", headers={"X-User-Id": "u0"})
    for i in range(1, n):
        service.JOBS[f"u{i}"] = service.Job(id=f"j{i}", created_at="", updated_at="",
                                            touched=time.time())
    t0 = time.perf_counter()
    for _ in range(20):
        c.get("/profiling", headers={"X-User-Id": "u0"})
    return (time.perf_counter() - t0) / 20

klein, gross = polling_dauer(20), polling_dauer(20000)
check(11, "GET /profiling scannt bei jedem Polling alle Jobs unter globaler Sperre",
      gross - klein > 0.0005,
      f"{klein*1000:.2f} ms bei 20 Jobs, {gross*1000:.2f} ms bei 20.000 Jobs "
      f"-> Mehraufwand {(gross-klein)*1000:.2f} ms/Polling; "
      f"bei 2s-Intervall {20000/2*(gross-klein)*100:.0f} % eines Kerns nur fuer den Sweep")

print("\n" + "=" * 78)
print(f"{sum(1 for r in R if r[2])}/{len(R)} Befunde am laufenden Dienst bestaetigt")
