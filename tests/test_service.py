# -*- coding: utf-8 -*-
"""End-to-End-Tests des eigenstaendigen Service (MockProvider, ohne Netz).

Deckt den Spec-Lebenszyklus ab: create -> attach -> submit -> poll ->
Single-Read-Result -> 410, plus den zustandslosen /embeddings-Endpunkt.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import service
from service import app

H = {"X-User-Id": "test-user"}


@pytest.fixture(autouse=True)
def clean_state():
    service.JOBS.clear()
    service.SUBMITS.clear()
    service.EMBEDS.clear()
    service._AUFRAEUMEN_FAELLIG.clear()
    service._sweep_due = float("inf")
    yield
    service.JOBS.clear()
    service.SUBMITS.clear()
    service.EMBEDS.clear()
    service._AUFRAEUMEN_FAELLIG.clear()
    service._sweep_due = float("inf")


def wait_done(client, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get("/profiling", headers=H).json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("Job wurde nicht fertig")


def test_full_job_lifecycle_with_single_read():
    with TestClient(app) as c:
        assert c.get("/profiling", headers=H).status_code == 404
        assert c.post("/profiling", headers=H).status_code == 201
        assert c.post("/profiling", headers=H).status_code == 409  # Singleton

        # leerer Job darf nicht submitted werden
        assert c.patch("/profiling", headers=H,
                       json={"status": "submitted"}).status_code == 409

        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("cv.txt",
                                   b"As a PhD student I work on natural language "
                                   b"processing and language models. Looking for "
                                   b"collaboration.", "text/plain")})
        assert r.status_code == 201 and r.json()["sha256"]

        assert c.patch("/profiling", headers=H,
                       json={"status": "submitted"}).status_code == 202  # wie das neighmash-Backend
        job = wait_done(c)
        assert job["status"] == "succeeded"

        # Single-Read: einmal 200, danach 410 (Pfad wie das neighmash-Backend: /profiling/result)
        res = c.get("/profiling/result", headers=H)
        assert res.status_code == 200
        prof = res.json()["profile"]
        assert prof["description"] and prof["embedding"]
        # Term-Vektoren sind voreingestellt aus (Speicher); sie kommen auf
        # Anforderung ueber /embeddings, siehe eigenen Test weiter unten.
        assert not prof.get("term_embeddings")
        assert c.get("/profiling/result", headers=H).status_code == 410

        assert c.delete("/profiling", headers=H).status_code == 204
        assert c.get("/profiling", headers=H).status_code == 404


def test_users_are_isolated():
    with TestClient(app) as c:
        assert c.post("/profiling", headers={"X-User-Id": "a"}).status_code == 201
        assert c.get("/profiling", headers={"X-User-Id": "b"}).status_code == 404


def test_document_rules(monkeypatch):
    # Grenze fuer den Test klein setzen: die Voreinstellung liegt bei 20 MB, und
    # es geht hier um das Verhalten an der Grenze, nicht um ihre Hoehe.
    monkeypatch.setattr(service, "MAX_DOC_BYTES", 5_000)
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        big = b"x" * (service.MAX_DOC_BYTES + 1)
        assert c.post("/profiling/documents", headers=H,
                      files={"file": ("big.txt", big, "text/plain")}).status_code == 413
        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("a.txt", b"Deep learning with transformers is my topic.",
                                   "text/plain")},
                   data={"metadata": '{"is_own": false}'})
        assert r.status_code == 201
        doc_id = r.json()["id"]
        assert c.delete(f"/profiling/documents/{doc_id}", headers=H).status_code == 204
        assert c.delete(f"/profiling/documents/{doc_id}", headers=H).status_code == 404


def test_urls_crud_and_draft_only():
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        r = c.post("/profiling/urls", headers=H,
                   json={"urls": [{"url": "https://example.org/about", "hint": "Homepage"}]})
        assert r.status_code == 201
        url_id = r.json()["added"][0]["id"]
        assert len(c.get("/profiling/urls", headers=H).json()) == 1
        assert c.delete(f"/profiling/urls/{url_id}", headers=H).status_code == 204


def test_embeddings_stateless_and_term_granularity():
    with TestClient(app) as c:
        assert c.post("/embeddings", headers=H,
                      json={"description": "kurz"}).status_code == 422
        r = c.post("/embeddings", headers=H, json={
            "description": "I work on machine learning and distributed systems.",
            "granularity": "terms", "dim": 64})
        assert r.status_code == 200
        body = r.json()
        assert len(body["embedding"]) == 64
        assert body["areas_of_interest"]
        assert all(len(t["embedding"]) == 64 for t in body["term_embeddings"])
        assert not service.JOBS                            # zustandslos


def test_embeddings_prueft_die_wertebereiche():
    """dim, quant und granularity nimmt der Endpunkt nur aus festen Wertebereichen.

    Vorher: Ein unbekanntes quant endete im Managed-Betrieb als HTTP 500, und die
    regelbasierte Referenz legte fuer ein beliebig grosses dim einen Vektor dieser
    Laenge an. Die Abweisung kommt als problem+json wie die uebrigen 422 hier.
    """
    text = "I work on machine learning and distributed systems."
    with TestClient(app) as c:
        for falsch in ({"quant": "foo"}, {"granularity": "alles"}, {"dim": -1},
                       {"dim": service.MAX_EMBED_DIM + 1}):
            r = c.post("/embeddings", headers=H, json={"description": text, **falsch})
            assert r.status_code == 422, falsch
            assert r.headers["content-type"].startswith("application/problem+json")
        # die Grenze selbst ist zulaessig
        assert c.post("/embeddings", headers=H, json={
            "description": text, "dim": service.MAX_EMBED_DIM}).status_code == 200


def test_repraesentation_des_embeddings_ist_einstellbar(monkeypatch):
    """FA8: welche Form in das mitgelieferte Embedding eingeht, ist Konfiguration.

    Voreingestellt der Profiltext (description), mit
    PROFILE_MINING_REPRESENTATION=structured die kanonische Serialisierung.
    """
    def ergebnis(c):
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"As a PhD student I work on natural language "
                                         b"processing. Looking for collaboration.",
                               "text/plain")})
        c.patch("/profiling", headers=H, json={"status": "submitted"})
        wait_done(c)
        prof = c.get("/profiling/result", headers=H).json()["profile"]
        c.delete("/profiling", headers=H)
        return prof

    p = service.make_provider()
    with TestClient(app) as c:
        prof = ergebnis(c)
        assert prof["embedding"] == p.embed(prof["description"])
        monkeypatch.setattr(service, "PROFILE_REPRESENTATION", "structured")
        prof = ergebnis(c)
        assert prof["embedding"] == p.embed(prof["canonical_text"])


def test_textdatei_in_latin1_kommt_lesbar_an():
    """Textdateien ohne gueltiges UTF-8 werden als Latin-1 gelesen, nicht mit
    Ersatzzeichen -- sonst verliert jeder Umlaut seine Bedeutung."""
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        text = "Übersicht über maschinelles Lernen für Datenbanken."
        assert c.post("/profiling/documents", headers=H,
                      files={"file": ("notiz.txt", text.encode("latin-1"),
                                      "text/plain")}).status_code == 201
        assert service.JOBS["test-user"].documents[0]["_text"] == text


def test_embeddings_grenze_je_identitaet(monkeypatch):
    """Auch /embeddings stoesst das Sprachmodell an und ist deshalb je Identitaet
    begrenzt; eine andere Identitaet ist davon nicht betroffen."""
    monkeypatch.setattr(service, "MAX_EMBEDS_PER_WINDOW", 2)
    body = {"description": "I work on machine learning and distributed systems."}
    with TestClient(app) as c:
        assert c.post("/embeddings", headers=H, json=body).status_code == 200
        assert c.post("/embeddings", headers=H, json=body).status_code == 200
        r = c.post("/embeddings", headers=H, json=body)
        assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
        assert c.post("/embeddings", headers={"X-User-Id": "andere"},
                      json=body).status_code == 200


def test_result_liefert_kanonische_serialisierung():
    """Studie 3.1 weist die strukturierte Form als bessere Embedding-Grundlage aus.

    Der vorgegebene Vertrag transportiert aber nur die natuerlichsprachliche
    description. Damit ein Client die bessere Variante nutzen kann, ohne die
    Feldreihenfolge selbst zu kennen, liefert das Ergebnis die fertig
    serialisierte Zeichenkette mit -- direkt an POST /embeddings uebergebbar.
    """
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt",
                               "Ich bin Doktorand und arbeite an Deep Learning mit "
                               "Transformer. Ich suche Forschungskooperationen.".encode(),
                               "text/plain")})
        c.patch("/profiling", headers=H, json={"status": "submitted"})
        wait_done(c)
        prof = c.get("/profiling/result", headers=H).json()["profile"]

        assert prof["canonical_text"], "canonical_text fehlt"
        assert prof["canonical_text"] != prof["description"]

        # Kennzeichnung nach Art. 50 Abs. 2 KI-VO: der regelbasierte MockProvider
        # erzeugt ohne Sprachmodell, der sprachmodellgestuetzte Provider kennzeichnet.
        assert prof["ai_generated"] is False
        assert service.ManagedLLMProvider.generative is True
        assert prof["matchable"] is True

        # Ein Profil ohne Forschungsfelder ist nicht matchfaehig: Seine Serialisierung
        # waere bei allen Betroffenen gleich, die Vektoren fielen zusammen.
        c.delete("/profiling", headers=H)
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("hobby.txt",
                               "Am Wochenende koche ich gern und gehe im Schwarzwald "
                               "wandern.".encode(), "text/plain")})
        c.patch("/profiling", headers=H, json={"status": "submitted"})
        wait_done(c)
        leer = c.get("/profiling/result", headers=H).json()["profile"]
        assert leer["structured"]["forschungsfelder"] == []
        assert leer["matchable"] is False
        for feld in prof["structured"]["forschungsfelder"]:
            assert feld.lower() in prof["canonical_text"].lower()

        # in beliebiger Groesse einbettbar -- das leistet der Ergebnisvektor nicht
        r = c.post("/embeddings", headers=H,
                   json={"description": prof["canonical_text"], "dim": 64,
                         "quant": "binary"})
        assert r.status_code == 200
        assert len(r.json()["embedding"]) == 64

        # Nach einer feldweisen Korrektur schickt die App das strukturierte Profil.
        # Der Dienst serialisiert es wie beim Mining, also gleicher Vektor, und
        # braucht dafuer kein Sprachmodell.
        aufrufe = []
        echt = service.MockProvider.extract_signals
        def zaehlend(self, *a, **kw):
            aufrufe.append(1)
            return echt(self, *a, **kw)
        service.MockProvider.extract_signals = zaehlend
        try:
            r = c.post("/embeddings", headers=H,
                       json={"structured": prof["structured"], "dim": 64, "quant": "binary"})
            vergleich = c.post("/embeddings", headers=H,
                               json={"description": prof["canonical_text"], "dim": 64,
                                     "quant": "binary"})
        finally:
            service.MockProvider.extract_signals = echt
        assert r.status_code == 200
        assert r.json()["embedding"] == vergleich.json()["embedding"]
        assert r.json()["areas_of_interest"] == prof["structured"]["forschungsfelder"]
        assert len(aufrufe) == 1, "strukturierter Aufruf darf kein Sprachmodell rufen"
        for falsch in ({}, {"description": prof["canonical_text"],
                            "structured": prof["structured"]},
                       {"structured": {"forschungsfelder": [" "]}}):
            assert c.post("/embeddings", headers=H, json=falsch).status_code == 422, falsch


# --------------------------------------------------------- Betrieb unter Last ---
# Die folgenden Tests sichern das Speicherverhalten in dem Szenario ab, das im
# Betrieb zuerst auftritt: Viele Teilnehmer laden gleichzeitig hoch, holen ihr
# Profil aber erst spaeter ab. Ohne diese Vorkehrungen waechst der Prozess
# unbegrenzt, weil sonst nur ein explizites DELETE aufraeumt.

def test_quelltexte_werden_nach_dem_mining_freigegeben():
    """Der Inhalt verschwindet nach der Verarbeitung, die Metadaten bleiben.

    Die Zusage lautet, dass Quellen nur fuer die Dauer des Job-Lebenszyklus
    vorgehalten werden. Das darf nicht davon abhaengen, ob der Client DELETE
    aufruft -- im Regelfall tut er es nicht.
    """
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt",
                               "Ich bin Doktorand und arbeite an Deep Learning mit Transformer. "
                               "Ich suche Forschungskooperationen.".encode(),
                               "text/plain")})
        job = service.JOBS["test-user"]
        assert job.documents[0]["_text"], "vor dem Mining muss der Text vorliegen"

        c.patch("/profiling", headers=H, json={"status": "submitted"})
        wait_done(c)

        assert job.documents[0]["_text"] == "", "Quelltext nach dem Mining nicht freigegeben"
        docs = c.get("/profiling/documents", headers=H).json()
        assert len(docs) == 1 and docs[0]["filename"] == "cv.txt"
        assert docs[0]["size"] > 0 and docs[0]["sha256"]


def test_dokumentanzahl_je_job_begrenzt():
    """Ohne Obergrenze bindet ein einzelner Nutzer beliebig viel Speicher."""
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        for i in range(service.MAX_DOCS_PER_JOB):
            r = c.post("/profiling/documents", headers=H,
                       files={"file": (f"d{i}.txt", b"Deep Learning.", "text/plain")})
            assert r.status_code == 201, f"Dokument {i} unerwartet abgelehnt"
        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("zuviel.txt", b"Deep Learning.", "text/plain")})
        assert r.status_code == 409
        assert str(service.MAX_DOCS_PER_JOB) in r.json()["detail"]


def test_urlanzahl_je_job_begrenzt():
    """Ohne Obergrenze nimmt ein einzelner Aufruf beliebig viele URLs an; der
    Worker haette sie sequenziell mit je URL_FETCH_TIMEOUT_S abgearbeitet und den
    Dienst damit zum Anfrage-Verstaerker gegen Dritte gemacht."""
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        zuviel = [{"url": f"https://example.org/{i}"}
                  for i in range(service.MAX_URLS_PER_JOB + 1)]
        r = c.post("/profiling/urls", headers=H, json={"urls": zuviel})
        assert r.status_code == 409
        assert str(service.MAX_URLS_PER_JOB) in r.json()["detail"]
        assert c.get("/profiling/urls", headers=H).json() == []
        # Die Grenze zaehlt kumulativ, nicht je Aufruf
        erlaubt = [{"url": f"https://example.org/{i}"} for i in range(service.MAX_URLS_PER_JOB)]
        assert c.post("/profiling/urls", headers=H, json={"urls": erlaubt}).status_code == 201
        assert c.post("/profiling/urls", headers=H,
                      json={"urls": [{"url": "https://example.org/x"}]}).status_code == 409


def test_defektes_pdf_ist_client_fehler():
    """pypdf-Ausnahmen ausser ImportError liefen in einen HTTP 500. Eine nicht
    lesbare Datei ist ein Eingabefehler des Clients, kein Serverfehler."""
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("kaputt.pdf", b"%PDF-1.4\nkein gueltiges PDF" + b"\x00" * 50,
                                   "application/pdf")})
        assert r.status_code == 400
        assert c.get("/profiling/documents", headers=H).json() == []


def test_dateigrenze_passt_zum_anwendungsfall():
    """Die Bytegrenze muss die Quellen der Zielgruppe zulassen.

    Fachartikel mit Abbildungen liegen bei 2 bis 10 MB, Dissertationen darueber.
    Eine Grenze von 2 MB haette sie ausgeschlossen. Dass sie steigen darf, ohne die
    Kosten zu erhoehen, liegt an der zweiten Ebene: die Zahl der Modellaufrufe
    haengt an MAX_EXTRACTED_CHARS, nicht an der Dateigroesse.
    """
    assert service.MAX_DOC_BYTES >= 10_000_000
    assert service.MAX_EXTRACTED_CHARS == 400_000


def test_upload_grenze_ohne_vollstaendiges_einlesen(monkeypatch):
    """Die Groessengrenze griff erst nach 'await file.read()', also nachdem der
    komplette Body im Speicher lag. read_limited() bricht darueber hinaus ab."""
    monkeypatch.setattr(service, "MAX_DOC_BYTES", 5_000)
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        zu_gross = b"a" * (service.MAX_DOC_BYTES + service.UPLOAD_READ_CHUNK)
        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("gross.txt", zu_gross, "text/plain")})
        assert r.status_code == 413
        # genau an der Grenze muss es noch durchgehen
        gerade_so = b"Deep Learning. " * (service.MAX_DOC_BYTES // 15)
        assert len(gerade_so) <= service.MAX_DOC_BYTES
        assert c.post("/profiling/documents", headers=H,
                      files={"file": ("passt.txt", gerade_so, "text/plain")}).status_code == 201


def test_fehlerdetails_gelangen_nicht_zum_client(monkeypatch):
    """job.stage trug den rohen Ausnahmetext -- einschliesslich der bis zu 500
    Zeichen Anbieterantwort, die ManagedLLMProvider._post mitfuehrt."""
    import provider as prov
    geheim = "invalid api key sk-abc123"

    def boom(self, chunk, lang):
        raise RuntimeError(f"uni-API: HTTP 401. Antwort: {{'error':'{geheim}'}}")

    monkeypatch.setattr(prov.MockProvider, "extract_signals", boom)
    service.reset_providers()
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep Learning und Transformer, Doktorand.",
                               "text/plain")})
        c.patch("/profiling", headers=H, json={"status": "submitted"})
        j = wait_done(c)
    assert j["status"] == "failed"
    assert geheim not in (j["stage"] or "")
    assert j["stage"] == "error"
    service.reset_providers()


def test_provider_wird_prozessweit_wiederverwendet():
    """make_provider() erzeugte je Anfrage eine neue Instanz. Im Managed-Betrieb
    lud damit jede Anfrage ihr eigenes Embedding-Modell (~0,5 GB je Kopie)."""
    service.reset_providers()
    assert service.make_provider() is service.make_provider()
    service.reset_providers()
    assert service.make_provider() is not None


def test_dienst_fordert_antwortschema_an(monkeypatch):
    """Der Dienst schaltet das Antwortschema des Signalschritts ein (Klasse: aus,
    damit die Studien reproduzierbar bleiben); PROFILE_MINING_RESPONSE_SCHEMA=0
    schaltet es ab."""
    monkeypatch.setenv("PROFILE_MINING_PROVIDER", "uni")
    monkeypatch.setenv("LLM_BASE_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "testmodell")
    service.reset_providers()
    assert service.make_provider().response_schema is True
    monkeypatch.setenv("PROFILE_MINING_RESPONSE_SCHEMA", "0")
    service.reset_providers()
    assert service.make_provider().response_schema is False
    service.reset_providers()


def test_job_verfaellt_nach_ttl(monkeypatch):
    """Der Job-TTL raeumt auch dann auf, wenn der Client nie DELETE aufruft.

    Die Uhr ist injizierbar, damit der Verfall ohne Wartezeit pruefbar ist
    (dasselbe Muster wie beim Result-Cache).
    """
    jetzt = [1000.0]
    monkeypatch.setattr(service, "CLOCK", lambda: jetzt[0])
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep Learning.", "text/plain")})
        assert "test-user" in service.JOBS

        jetzt[0] += service.JOB_TTL_S - 1            # kurz vor Ablauf
        assert c.get("/profiling", headers=H).status_code == 200

        jetzt[0] += 2                                # jetzt abgelaufen
        assert c.get("/profiling", headers=H).status_code == 404
        assert "test-user" not in service.JOBS, "Job samt Quellen nicht freigegeben"


def test_ergebnis_verschwindet_mit_dem_job(monkeypatch):
    """Ein nicht abgeholtes Ergebnis gehoert zum Job und verschwindet mit ihm.

    Frueher entfernte DELETE /profiling nur den Job; das Ergebnis blieb im
    Result-Cache stehen, und der raeumte nur bei einem Lesezugriff ab. Geprueft
    wird deshalb der Speicher selbst und nicht die Antwort -- abrufbar war das
    Ergebnis in beiden Faellen schon vorher nicht mehr.
    """
    def fertiger_job(c):
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep learning and transformers.", "text/plain")})
        c.patch("/profiling", headers=H, json={"status": "submitted"})
        assert wait_done(c)["status"] == "succeeded"
        return service.JOBS["test-user"].result_token

    with TestClient(app) as c:
        # 1) DELETE verwirft das nicht abgeholte Ergebnis
        token = fertiger_job(c)
        assert token in service.RESULTS._store
        assert c.delete("/profiling", headers=H).status_code == 204
        assert token not in service.RESULTS._store, "Ergebnis ueberlebt DELETE"

        # 2) Nach Fristablauf raeumt schon das Statuspolling, ohne dass jemand liest
        token = fertiger_job(c)
        jetzt = [time.time() + service.RESULT_TTL_S + 1]
        monkeypatch.setattr(service.RESULTS, "_clock", lambda: jetzt[0])
        assert c.get("/profiling", headers=H).status_code == 200
        assert token not in service.RESULTS._store, "abgelaufenes Ergebnis bleibt liegen"
        assert c.get("/profiling/result", headers=H).status_code == 410
        assert c.delete("/profiling", headers=H).status_code == 204

        # 3) Wird der Job waehrend des Minings geloescht, bricht das Mining nach dem
        #    laufenden Chunk ab, statt alle weiteren Modellaufrufe auszufuehren
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep learning and transformers. " * 200,
                               "text/plain")})
        vorher = set(service.RESULTS._store)
        aufrufe = []
        echt_extrahieren = service.MockProvider.extract_signals

        def extrahieren_und_loeschen(self, chunk, lang):
            aufrufe.append(chunk)
            service.JOBS.pop("test-user", None)          # DELETE mitten im Mining
            return echt_extrahieren(self, chunk, lang)

        with monkeypatch.context() as m:
            m.setattr(service.MockProvider, "extract_signals", extrahieren_und_loeschen)
            service.run_job("test-user")                 # synchron statt im Thread
        assert len(aufrufe) == 1, f"Mining rechnet nach DELETE weiter ({len(aufrufe)} Chunks)"
        assert set(service.RESULTS._store) == vorher, "verwaistes Ergebnis nach DELETE"

        # 4) Faellt das Loeschen hinter den letzten Modellaufruf, verwirft die
        #    Schlusspruefung unter der Sperre das Ergebnis
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep learning and transformers.", "text/plain")})
        echt_einbetten = service.MockProvider.embed

        def einbetten_und_loeschen(self, *a, **kw):
            service.JOBS.pop("test-user", None)          # DELETE beim Einbetten
            return echt_einbetten(self, *a, **kw)

        with monkeypatch.context() as m:
            m.setattr(service.MockProvider, "embed", einbetten_und_loeschen)
            service.run_job("test-user")
        assert set(service.RESULTS._store) == vorher, "verwaistes Ergebnis nach DELETE"


def test_job_ueberlebt_das_result():
    """Der Job-TTL muss laenger sein als der Result-TTL.

    Sonst antwortet ein zweiter Abruf mit 404 statt mit dem vertraglich
    vorgesehenen 410 Gone.
    """
    assert service.JOB_TTL_S > service.RESULT_TTL_S


def test_result_traegt_ohne_anforderung_keine_term_vektoren():
    """Term-Vektoren machen den Grossteil des Result-Eintrags aus.

    Der Vertrag verlangt sie nicht; wer sie braucht, fordert sie ueber
    /embeddings an. Voreingestellt bleiben sie deshalb aus.
    """
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt",
                               "Ich bin Doktorand und arbeite an Deep Learning mit Transformer. "
                               "Ich suche Forschungskooperationen.".encode(),
                               "text/plain")})
        c.patch("/profiling", headers=H, json={"status": "submitted"})
        wait_done(c)
        prof = c.get("/profiling/result", headers=H).json()["profile"]
        assert prof.get("term_embeddings") in (None, [])
        assert prof["embedding"], "der Profilvektor gehoert weiterhin ins Ergebnis"
        r = c.post("/embeddings", headers=H,
                   json={"description": prof["canonical_text"], "granularity": "terms"})
        assert r.status_code == 200 and r.json()["term_embeddings"]


# --------------------------------------------------------- URL-Zielpruefung ---
# Serverseitige Anfragefaelschung (SSRF): Der Dienst darf nicht als Mittelsmann
# fuer Adressen dienen, die von aussen nicht erreichbar sind. Geprueft wird das
# Ergebnis der Namensaufloesung, nicht der Hostname -- sonst genuegte ein
# A-Record auf 127.0.0.1, um die Sperrliste zu umgehen.

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/admin",
    "http://localhost/admin",
    "http://169.254.169.254/latest/meta-data/",   # Cloud-Metadatendienst
    "http://10.1.2.3/internal",
    "http://192.168.0.5/",
    "http://172.16.5.5/",
    "http://100.64.0.1/",                          # CGNAT
    "http://0.0.0.0/",
    "http://[::1]/",
    "http://[::ffff:127.0.0.1]/",                  # IPv4-mapped Loopback
])
def test_gesperrte_ziele_werden_abgewiesen(url):
    with pytest.raises(ValueError):
        service.assert_url_allowed(url)


@pytest.mark.parametrize("url", ["ftp://example.org/x", "file:///etc/passwd", "gopher://x/"])
def test_nur_http_und_https(url):
    with pytest.raises(ValueError):
        service.assert_url_allowed(url)


def test_oeffentliche_adresse_wird_zugelassen():
    service.assert_url_allowed("https://8.8.8.8/")   # loest ohne DNS auf


def test_positivliste_greift(monkeypatch):
    monkeypatch.setattr(service, "_ALLOWED_HOSTS", ["example.org"])
    with pytest.raises(ValueError, match="URL_ALLOWED_HOSTS"):
        service.assert_url_allowed("https://8.8.8.8/")


def test_weiterleitung_in_gesperrten_bereich_wird_geprueft():
    """Der Redirect-Handler prueft jedes Ziel erneut; sonst genuegte eine
    oeffentliche URL, die per 302 auf den Metadatendienst zeigt."""
    handler = service._GuardedRedirectHandler()
    with pytest.raises(ValueError):
        handler.redirect_request(None, None, 302, "Found", {},
                                 "http://169.254.169.254/latest/meta-data/")


# ------------------------------------------------------------------------------
# Regressionstests zur Sicherheitsanalyse vom 23.08.2026
# ------------------------------------------------------------------------------

def test_extrahierter_text_wird_gekappt():
    """Die Bytegrenze des Uploads begrenzt den extrahierten Text nicht.

    Der Aufwand der Pipeline haengt an der Zeichenzahl NACH der Dekompression,
    nicht an der Groesse des Bodys: ein kleiner komprimierter Strom erzeugt
    beliebig viele Abschnitte und damit beliebig viele Modellaufrufe.
    """
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        lang = b"Deep Learning. " * (service.MAX_EXTRACTED_CHARS // 5)
        assert len(lang) < service.MAX_DOC_BYTES, "soll die Bytegrenze gerade nicht reissen"
        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("lang.txt", lang, "text/plain")})
        assert r.status_code == 201
        doc = service.JOBS["test-user"].documents[0]
        assert len(doc["_text"]) == service.MAX_EXTRACTED_CHARS
        assert doc["_truncated"] is True
        # Die Kappung ist intern: die Antwort traegt weiter nur Metadaten.
        assert "_truncated" not in r.json()


def test_pdf_extraktion_bricht_bei_erreichter_grenze_ab(monkeypatch):
    """Erst sammeln, dann kappen wuerde den entpackten Text vollstaendig im
    Speicher halten -- dieselbe Reihenfolge, die beim Upload schon einmal falsch
    war. Der Extraktor muss die Seiten abbrechen, sobald die Grenze faellt."""
    gelesen = []

    class Seite:
        def extract_text(self):
            gelesen.append(1)
            return "x" * 100_000

    class Reader:
        def __init__(self, *a, **k):
            self.pages = [Seite() for _ in range(1000)]

    import pypdf
    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        r = c.post("/profiling/documents", headers=H,
                   files={"file": ("bombe.pdf", b"%PDF-1.4 egal", "application/pdf")})
        assert r.status_code == 201
        # 400.000 Zeichen / 100.000 je Seite -> nach der fuenften Seite ist Schluss
        assert len(gelesen) <= 6, f"{len(gelesen)} Seiten gelesen statt frueh abzubrechen"
        assert len(service.JOBS["test-user"].documents[0]["_text"]) == service.MAX_EXTRACTED_CHARS


def test_embeddings_prueft_die_identitaet(monkeypatch):
    """/embeddings war der einzige Endpunkt ohne get_user(): ein Aufruf ohne jede
    Kennung erreichte die kostenpflichtige Anbieter-API."""
    monkeypatch.setattr(service, "STRICT_AUTH", True)
    with TestClient(app) as c:
        gut = {"description": "Ich forsche zu verteilten Systemen und Konsens."}
        assert c.post("/embeddings", json=gut).status_code == 401
        assert c.post("/embeddings", headers=H, json=gut).status_code == 200


def test_strict_auth_ersetzt_die_sammelidentitaet(monkeypatch):
    """Ohne den Schalter beantwortet der Dienst einen Aufruf ohne Kennung als
    'demo-user'; mehrere Aufrufer teilen sich dann denselben Singleton-Job."""
    assert service.get_user.__module__ == "service"
    monkeypatch.setattr(service, "STRICT_AUTH", True)
    with TestClient(app) as c:
        assert c.get("/profiling").status_code == 401
        assert c.post("/profiling").status_code == 401
    monkeypatch.setattr(service, "STRICT_AUTH", False)
    with TestClient(app) as c:
        assert c.get("/profiling").status_code == 404      # demo-user, kein Job



def _bearer(payload) -> dict:
    """Unsigniertes Token im JWT-Format. Der Dienst liest nur den Payload; die
    Signatur prueft im Verbund die Instanz davor."""
    import base64, json
    teil = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return {"Authorization": "Bearer " + teil({"alg": "none"}) + "." + teil(payload) + ".sig"}


def test_kennung_aus_dem_bearer_token(monkeypatch):
    """Der Weg der App: Die Kennung kommt aus dem sub-Claim des Tokens, und zwei
    Tokens mit verschiedenem sub erhalten je einen eigenen Job."""
    monkeypatch.setattr(service, "STRICT_AUTH", True)
    with TestClient(app) as c:
        a, b = _bearer({"sub": "alice"}), _bearer({"sub": "bob"})
        assert c.post("/profiling", headers=a).status_code == 201
        assert c.post("/profiling", headers=b).status_code == 201
        assert c.post("/profiling", headers=a).status_code == 409    # Singleton je sub
        assert set(service.JOBS) == {"alice", "bob"}
        assert (c.get("/profiling", headers=a).json()["id"]
                != c.get("/profiling", headers=b).json()["id"])


@pytest.mark.parametrize("payload", [{"iss": "keycloak"}, {"sub": ""}, ["kein", "objekt"]],
                         ids=["ohne-sub", "leerer-sub", "kein-objekt"])
def test_token_ohne_sub_ist_keine_kennung(monkeypatch, payload):
    """Ein Token ohne verwertbaren sub-Claim fiel frueher auf 'demo-user' zurueck,
    auch unter STRICT_AUTH; zwei solche Aufrufer teilten sich dann einen Job."""
    monkeypatch.setattr(service, "STRICT_AUTH", True)
    with TestClient(app) as c:
        assert c.post("/profiling", headers=_bearer(payload)).status_code == 401
        assert c.post("/embeddings", headers=_bearer(payload),
                      json={"description": "Ich forsche zu verteilten Systemen."}
                      ).status_code == 401
    assert "demo-user" not in service.JOBS

def test_embeddings_hat_laengen_und_termgrenze(monkeypatch):
    """Der Endpunkt ist zustandslos, aber nicht kostenlos: die Textlaenge und die
    Zahl der Terme bestimmen unmittelbar die Zahl der Anbieteraufrufe."""
    with TestClient(app) as c:
        zu_lang = "Deep Learning und verteilte Systeme. " * (service.MAX_EMBED_CHARS // 10)
        assert c.post("/embeddings", headers=H,
                      json={"description": zu_lang}).status_code == 413

        monkeypatch.setattr(service, "MAX_EMBED_TERMS", 2)
        r = c.post("/embeddings", headers=H, json={
            "description": "Ich arbeite zu Deep Learning, Computer Vision, "
                           "verteilten Systemen und Datenbanken als Doktorand.",
            "dim": 32, "granularity": "terms"})
        assert r.status_code == 200
        assert len(r.json()["term_embeddings"]) <= 2


def test_sweep_scannt_nur_wenn_ein_verfall_moeglich_ist(monkeypatch):
    """Der Scan lief unter der globalen Sperre bei jedem Polling. _sweep_due ist
    eine exakte Schranke, kein Zeitraster: der Verfallszeitpunkt selbst darf sich
    dadurch nicht verschieben (das prueft test_job_verfaellt_nach_ttl)."""
    class ZaehlDict(dict):
        """Zaehlt, wie oft sweep_jobs() tatsaechlich ueber alle Jobs laeuft."""
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    jetzt = [1000.0]
    monkeypatch.setattr(service, "CLOCK", lambda: jetzt[0])
    monkeypatch.setattr(service, "JOBS", ZaehlDict())
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        assert service._sweep_due == 1000.0 + service.JOB_TTL_S

        service.JOBS.scans = 0
        c.get("/profiling", headers=H)
        assert service.JOBS.scans == 0, "vor dem fruehesten Verfall darf kein Scan laufen"

        jetzt[0] += service.JOB_TTL_S + 1
        c.get("/profiling", headers=H)
        assert service.JOBS.scans == 1, "nach Ablauf muss genau ein Scan laufen"
        assert "test-user" not in service.JOBS


def test_ratengrenze_je_identitaet(monkeypatch):
    """Alle uebrigen Grenzen des Dienstes zaehlen je Job. Ohne eine Grenze je
    Identitaet koennte ein Aufrufer beliebig viele Jobs nacheinander einreichen
    und damit Laufzeit und Anbieterkontingent binden."""
    monkeypatch.setattr(service, "MAX_JOBS_PER_WINDOW", 2)
    with TestClient(app) as c:
        for i in range(2):
            c.post("/profiling", headers=H)
            c.post("/profiling/documents", headers=H,
                   files={"file": (f"cv{i}.txt", b"Deep Learning.", "text/plain")})
            assert c.patch("/profiling", headers=H,
                           json={"status": "submitted"}).status_code == 202
            wait_done(c)
            c.delete("/profiling", headers=H)

        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv3.txt", b"Deep Learning.", "text/plain")})
        r = c.patch("/profiling", headers=H, json={"status": "submitted"})
        assert r.status_code == 429
        assert r.headers.get("Retry-After")
        assert r.json()["title"] == "Too Many Requests"
        # Der Job bleibt im Entwurf und wird nicht verarbeitet.
        assert service.JOBS["test-user"].status == "draft"

        # Eine andere Identitaet ist davon unberuehrt.
        H2 = {"X-User-Id": "zweiter-nutzer"}
        c.post("/profiling", headers=H2)
        c.post("/profiling/documents", headers=H2,
               files={"file": ("cv.txt", b"Deep Learning.", "text/plain")})
        assert c.patch("/profiling", headers=H2,
                       json={"status": "submitted"}).status_code == 202


def test_ratenfenster_verfaellt(monkeypatch):
    """Die Grenze ist ein gleitendes Fenster, keine harte Lebenszeitquote.
    Identitaeten ohne Aufruf im Fenster werden dabei aus dem Speicher entfernt."""
    jetzt = [1000.0]
    monkeypatch.setattr(service, "CLOCK", lambda: jetzt[0])
    monkeypatch.setattr(service, "MAX_JOBS_PER_WINDOW", 1)
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep Learning.", "text/plain")})
        assert c.patch("/profiling", headers=H,
                       json={"status": "submitted"}).status_code == 202
        wait_done(c)
        c.delete("/profiling", headers=H)

        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("cv.txt", b"Deep Learning.", "text/plain")})
        assert c.patch("/profiling", headers=H,
                       json={"status": "submitted"}).status_code == 429
        service.SUBMITS["laengst-still"] = [jetzt[0]]

        jetzt[0] += service.RATE_WINDOW_S + 1
        assert c.patch("/profiling", headers=H,
                       json={"status": "submitted"}).status_code == 202
        assert service.SUBMITS == {"test-user": [jetzt[0]]}


def test_globale_jobobergrenze(monkeypatch):
    """Je Job sind bis zu 20 Dokumente à 2 MB moeglich; ohne Obergrenze waechst
    der Speicherbedarf des Prozesses mit der Zahl offener Entwuerfe."""
    monkeypatch.setattr(service, "MAX_JOBS_TOTAL", 3)
    with TestClient(app) as c:
        for i in range(3):
            assert c.post("/profiling",
                          headers={"X-User-Id": f"u{i}"}).status_code == 201
        r = c.post("/profiling", headers={"X-User-Id": "u3"})
        assert r.status_code == 503
        assert r.json()["title"] == "Service Unavailable"


def test_ohne_kennung_401_in_der_voreinstellung():
    """Die Voreinstellung ist die sichere: Ein Aufruf ohne verwertbare Kennung
    wird abgewiesen, statt auf eine Sammelidentitaet abgebildet zu werden."""
    assert service.STRICT_AUTH is True
    with TestClient(app) as c:
        assert c.get("/profiling").status_code == 401
        assert c.post("/embeddings",
                      json={"description": "Verteilte Systeme und Konsens."}).status_code == 401
        # Mit Kennung laeuft derselbe Aufruf durch.
        assert c.post("/embeddings", headers=H,
                      json={"description": "Verteilte Systeme und Konsens."}).status_code == 200


def test_ergebnis_und_fehler_sind_nicht_zwischenspeicherbar():
    """Single-Read endet sonst an der Serverkante.

    Anlass: Ein Browser hielt die 410-Antwort im Cache und beantwortete jeden
    weiteren Abruf daraus -- ohne Anfrage an den Dienst. Schwerer waere derselbe
    Fall bei der 200er-Antwort: Sie traegt das vollstaendige Profil, und der
    serverseitige Loeschvorgang naehme dem Cache nichts weg.
    """
    with TestClient(app) as c:
        c.post("/profiling", headers=H)
        c.post("/profiling/documents", headers=H,
               files={"file": ("a.txt", b"Deep learning with transformers is my topic.",
                               "text/plain")})
        c.patch("/profiling", json={"status": "submitted"}, headers=H)
        erster = c.get("/profiling/result", headers=H)
        assert erster.status_code == 200
        assert erster.headers["cache-control"] == "no-store"
        zweiter = c.get("/profiling/result", headers=H)
        assert zweiter.status_code == 410
        assert zweiter.headers["cache-control"] == "no-store"
        # auch der Zustand des Jobs ist nichts, was ein Cache aufheben darf
        assert c.get("/profiling", headers=H).headers["cache-control"] == "no-store"
