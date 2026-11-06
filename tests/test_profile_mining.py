# -*- coding: utf-8 -*-
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provider import MockProvider, Signal, ManagedLLMProvider, extract_json_object
from pipeline import ProfileMiner, Source, chunk_text, detect_lang, _struct_to_text
from embedding_compression import matryoshka_truncate, quantize, cosine, hamming_sim, l2norm
from cache import ResultCache, SingleReadExpired
import pytest


# ---------- Provider ----------
def test_extract_signals_have_evidence_and_kinds():
    p = MockProvider()
    txt = "Ich bin Doktorand und arbeite an Deep Learning mit Transformer. Ich suche Forschungskooperationen."
    sig = p.extract_signals(txt, "de")
    assert sig
    assert all(isinstance(s, Signal) and s.evidence for s in sig)
    kinds = {s.kind for s in sig}
    assert "field" in kinds and "role" in kinds

def test_embed_is_unit_norm_and_dim():
    p = MockProvider()
    v = p.embed("machine learning and databases", dim=128)
    assert len(v) == 128
    assert abs(math.sqrt(sum(x*x for x in v)) - 1.0) < 1e-6


def test_mock_wendet_kuerzung_und_quantisierung_an():
    """Die regelbasierte Referenz nutzt dieselben Stufen wie der Managed-Pfad.

    Vorher bestimmte dim dort die Breite der Hash-Tabelle, und quant wurde
    ignoriert -- mit der voreingestellten Konfiguration lieferte /embeddings
    dann trotz quant="binary" Gleitkommazahlen.
    """
    p, text = MockProvider(), "machine learning and distributed systems"
    voll = p.embed(text)
    assert len(voll) == MockProvider.NATIVE_DIM
    assert p.embed(text, dim=64) == matryoshka_truncate(voll, 64)
    q = p.embed(text, dim=64, quant="int8")
    assert len(q) == 64 and all(isinstance(x, int) and -127 <= x <= 127 for x in q)
    b = p.embed(text, dim=64, quant="binary")
    assert len(b) == 64 and set(b) <= {0, 1}


# ---------- Chunking ----------
def test_chunking_respects_size():
    long = ("Satz eins. " * 400)
    chunks = chunk_text(long, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 360 for c in chunks)


# ---------- Pipeline ----------
def test_mine_produces_both_representations_and_evidence():
    miner = ProfileMiner(MockProvider())
    src = [Source(id="d1", text="As a PhD student I work on natural language processing and language models. Looking for collaboration.", is_own=True)]
    prof = miner.mine(src)
    assert prof.structured["forschungsfelder"]
    assert prof.text and isinstance(prof.text, str)
    assert len(prof.embedding) == 256
    assert all(s["evidence"] for s in prof.signals)
    sal = [s["salience"] for s in prof.signals]
    assert sal == sorted(sal, reverse=True)

def test_authority_lowers_read_only_sources():
    # verify=False isoliert die Aggregation: geprueft wird die Gewichtung selbst,
    # nicht ob das Ergebnis anschliessend die Verifikationsschwelle passiert.
    miner = ProfileMiner(MockProvider(), cfg={"verify": False})
    own = miner.mine([Source(id="a", text="Deep learning and transformers.", is_own=True)]).signals
    read = miner.mine([Source(id="b", text="Deep learning and transformers.", is_own=False)]).signals
    s_own = next(s["salience"] for s in own if s["value"] == "dl")
    s_read = next(s["salience"] for s in read if s["value"] == "dl")
    assert s_own > s_read


def test_gelesene_quelle_bleibt_erhalten():
    """Ein Thema aus einer nur gelesenen Quelle gehoert ins Profil.

    In einer Matching-App laedt niemand versehentlich ein Dokument hoch: Der
    Upload ist die Interessensbekundung, unabhaengig davon, wer den Text
    verfasst hat. Die Autoritaetsgewichtung stuft solche Themen zurueck, sie
    schliesst sie nicht aus.
    """
    miner = ProfileMiner(MockProvider())          # Default-Konfiguration
    prof = miner.mine([Source(id="gelesen", text="Deep learning and transformers.",
                              is_own=False)])
    assert any(s["value"] == "dl" for s in prof.signals)
    assert "Deep Learning" in prof.structured["forschungsfelder"]


def test_gelesenes_thema_rangiert_hinter_eigenem():
    """Der eigentliche Anspruch an die Gewichtung: Reihenfolge, nicht Ausschluss.

    Ein selbst bearbeitetes Thema muss vor einem nur gelesenen liegen, damit das
    Matching sich primaer auf die eigene Arbeit stuetzt.
    """
    miner = ProfileMiner(MockProvider())
    prof = miner.mine([
        Source(id="eigen", text="My work is on computer vision and object detection.",
               is_own=True),
        Source(id="gelesen", text="Read: a survey on deep learning and transformers.",
               is_own=False),
    ])
    sal = {s["value"]: s["salience"] for s in prof.signals if s["kind"] == "field"}
    assert "cv" in sal and "dl" in sal, f"beide Themen erwartet, erhalten: {sal}"
    assert sal["cv"] > sal["dl"]
    felder = prof.structured["forschungsfelder"]
    assert felder.index("Computer Vision") < felder.index("Deep Learning")

def test_multi_source_bonus_increases_salience():
    miner = ProfileMiner(MockProvider())
    one = miner.mine([Source(id="a", text="Deep learning with transformers is my topic.")]).signals
    two = miner.mine([Source(id="a", text="Deep learning with transformers is my topic."),
                      Source(id="b", text="My research focuses on deep learning methods.")]).signals
    s1 = next(s["salience"] for s in one if s["value"] == "dl")
    s2 = next(s["salience"] for s in two if s["value"] == "dl")
    assert s2 > s1
    assert next(s["n_sources"] for s in two if s["value"] == "dl") == 2

def test_too_short_sources_are_ignored():
    # Ingestion verwirft Quellen unter 30 Zeichen
    prof = ProfileMiner(MockProvider()).mine([Source(id="a", text="Deep learning.")])
    assert prof.signals == []

def test_verification_drops_below_threshold():
    # Schwelle ueber Methoden-Konfidenz (0,7), unter Feld-Konfidenz (0,9):
    # Methoden-Signale muessen herausfallen, Felder bleiben.
    miner = ProfileMiner(MockProvider(), cfg={"min_salience": 0.8})
    prof = miner.mine([Source(id="a", text="Deep learning with backpropagation.")])
    kinds = {s["kind"] for s in prof.signals}
    assert "field" in kinds and "method" not in kinds

def test_structured_lists_are_capped():
    txt = ("machine learning, deep learning, natural language processing, computer vision, "
           "reinforcement learning, information retrieval, databases and distributed systems. "
           "I use transformer, backpropagation, attention, policy gradient, q-learning, "
           "slam, sensor fusion and sharding.")
    prof = ProfileMiner(MockProvider()).mine([Source(id="a", text=txt)])
    assert len(prof.structured["forschungsfelder"]) <= 5
    assert len(prof.structured["methoden"]) <= 6

def test_detect_lang():
    assert detect_lang("Ich bin Doktorand und arbeite an der Universität.") == "de"
    assert detect_lang("I am looking for research collaborations with the group.") == "en"

def test_struct_to_text_is_canonical_and_stable():
    s = {"forschungsfelder": ["NLP"], "methoden": ["attention"], "rolle": "PhD student", "sucht": ["mentorship"]}
    out1, out2 = _struct_to_text(s), _struct_to_text(dict(s))
    assert out1 == out2
    assert out1.index("fields:") < out1.index("methods:") < out1.index("role:") < out1.index("seeking:")


# ---------- Compression ----------
def test_matryoshka_truncate_normalized():
    v = l2norm([float(i) for i in range(256)])
    t = matryoshka_truncate(v, 64)
    assert len(t) == 64
    assert abs(math.sqrt(sum(x*x for x in t)) - 1.0) < 1e-6

def test_quantize_byte_sizes():
    v = l2norm([0.1]*256)
    _, b32 = quantize(v, "float32"); _, b8 = quantize(v, "int8"); _, bb = quantize(v, "binary")
    assert (b32, b8, bb) == (1024, 256, 32)

def test_cosine_identity_and_hamming():
    v = l2norm([0.2, -0.3, 0.5, 0.1])
    assert abs(cosine(v, v) - 1.0) < 1e-9
    assert hamming_sim([1,0,1,1], [1,0,1,1]) == 1.0
    assert hamming_sim([1,0,1,0], [0,1,0,1]) == 0.0


# ---------- ManagedLLMProvider (ohne Netz, _chat gemockt) ----------
def test_extract_json_object_tolerates_fences_and_prose():
    raw = ('Hier ist das Ergebnis:\n```json\n'
           '{"signals": [{"kind": "field", "value": "NLP", "evidence": "x", "confidence": 0.9}]}\n```')
    obj = extract_json_object(raw)
    assert obj["signals"][0]["value"] == "NLP"

def test_managed_provider_enforces_schema_and_evidence(monkeypatch):
    p = ManagedLLMProvider(vendor="openai")
    fake = ('{"signals": ['
            '{"kind": "field", "value": "Deep Learning", "evidence": "wir nutzen DL", "confidence": 1.7},'
            '{"kind": "method", "value": "attention", "evidence": "", "confidence": 0.8},'
            '{"kind": "bogus", "value": "x", "evidence": "y", "confidence": 0.5}]}')
    monkeypatch.setattr(p, "_chat", lambda prompt: fake)
    sig = p.extract_signals("chunk", "de")
    assert len(sig) == 1                       # ohne Beleg / falsches Schema -> verworfen
    assert sig[0].value == "deep learning"
    assert sig[0].confidence == 1.0            # Konfidenz auf [0,1] geklemmt

def test_managed_provider_rejects_unknown_vendor():
    with pytest.raises(ValueError):
        ManagedLLMProvider(vendor="vertex")

def test_pipeline_uses_provider_synthesis_if_available(monkeypatch):
    class SynthMock(MockProvider):
        def synthesize_text(self, structured):
            return "LLM-SYNTHESE"
    prof = ProfileMiner(SynthMock()).mine(
        [Source(id="a", text="As a PhD student I work on deep learning models.")])
    assert prof.text == "LLM-SYNTHESE"


# ---------- Result-Cache: Single-Read + TTL ----------
def test_single_read_then_gone():
    c = ResultCache(ttl_seconds=3600)
    tok = c.put({"profile": "x"})
    assert c.read_once(tok) == {"profile": "x"}
    with pytest.raises(SingleReadExpired):
        c.read_once(tok)

def test_ttl_expiry():
    clk = {"t": 1000.0}
    c = ResultCache(ttl_seconds=10, _clock=lambda: clk["t"])
    tok = c.put({"p": 1})
    clk["t"] = 1011.0
    assert not c.exists(tok)
    with pytest.raises(SingleReadExpired):
        c.read_once(tok)


# ---------- Synthese-Parser (Regression) ----------
# Der erzwungene JSON-Modus von OpenAI/Mistral laesst das Modell die Prosa der
# Synthese-Stufe in ein JSON-Objekt mit frei gewaehltem Schluessel packen. Ohne
# Entpacken landet die Huelle im Profiltext und damit im Embedding.
@pytest.mark.parametrize("raw,expected", [
    ('{"profiltext": "Schwerpunkt IT-Sicherheit."}', "Schwerpunkt IT-Sicherheit."),
    ('{"text": "Schwerpunkt Computer Vision."}', "Schwerpunkt Computer Vision."),
    ('{"zusammenfassung": "Kurzprofil Robotik."}', "Kurzprofil Robotik."),
    ('```json\n{"profiltext": "Mit Codezaun."}\n```', "Mit Codezaun."),
    ('{"rolle": "postdoc", "profiltext": "Der eigentliche Profiltext."}',
     "Der eigentliche Profiltext."),
])
def test_synthese_entpackt_json_huelle(raw, expected):
    assert ManagedLLMProvider._unwrap_prose(raw) == expected


@pytest.mark.parametrize("raw", [
    "Als Postdoc liegen die Schwerpunkte in IT-Sicherheit.",  # Uni-API: reine Prosa
    "{kaputt",                                                # unparsbar -> Rohtext
    "",
])
def test_synthese_laesst_nicht_json_unveraendert(raw):
    assert ManagedLLMProvider._unwrap_prose(raw) == raw.strip()


def test_profiltext_enthaelt_keine_json_huelle():
    """Ein synthetisierter Profiltext darf nie als JSON-Objekt beginnen."""
    for raw in ('{"profiltext": "A."}', '{"text": "B."}', "C."):
        assert not ManagedLLMProvider._unwrap_prose(raw).lstrip().startswith("{")


# ---------- Synthese: Wiederholung statt Jobverlust ----------
# Ein transienter 500er im LETZTEN Schritt verwarf frueher die gesamte vorher
# geleistete Extraktion. Die Docstring versprach eine "Wiederholungslogik des
# Aufrufers" -- pipeline.mine ruft die Synthese aber genau einmal auf.

def _provider_mit_antworten(monkeypatch, antworten):
    """Baut einen Provider, dessen _chat die Liste der Reihe nach abarbeitet.

    Eintraege duerfen Strings (Antwort) oder Exceptions (Fehler) sein.
    """
    import time
    monkeypatch.setattr(time, "sleep", lambda _s: None)   # Backoff nicht abwarten
    p = ManagedLLMProvider(vendor="uni", model="testmodell", base_url="http://test/v1")
    folge = list(antworten)

    def fake_chat(prompt):
        naechste = folge.pop(0)
        if isinstance(naechste, Exception):
            raise naechste
        return naechste

    monkeypatch.setattr(p, "_chat", fake_chat)
    return p


def _transient(text="uni-API: HTTP 500. Antwort: "):
    e = RuntimeError(text)
    e.transient = True
    return e


def test_synthese_wiederholt_transiente_fehler(monkeypatch):
    p = _provider_mit_antworten(monkeypatch, [
        _transient(), _transient(),
        "Eine Doktorandin arbeitet an Informationssuche und sucht Kooperationen.",
    ])
    text = p.synthesize_text({"forschungsfelder": ["Information Retrieval"]})
    assert text.startswith("Eine Doktorandin")



def test_zeitueberschreitung_und_verbindungsabbruch_werden_wiederholt(monkeypatch):
    """Timeouts und Verbindungsfehler entstehen in _post. Die Wiederholung prueft
    e.transient -- diese Markierung fehlte dort, und der Job scheiterte beim
    ersten Aussetzer des Dienstes. Der Test geht bewusst durch _post statt _chat
    zu ersetzen, weil die Luecke genau dort lag."""
    import json, time, urllib.error, urllib.request
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    antwort = {"choices": [{"message": {"content": json.dumps({"signals": [
        {"kind": "field", "value": "Information Retrieval",
         "evidence": "arbeite an Information Retrieval", "confidence": 0.9}]})}}]}

    class Antwort:
        def __init__(self, daten): self.daten = daten
        def read(self): return json.dumps(self.daten).encode("utf-8")
        def __enter__(self): return self
        def __exit__(self, *a): return False

    folge = [TimeoutError(), urllib.error.URLError("connection reset"), Antwort(antwort)]

    def fake_urlopen(req, timeout=None):
        naechste = folge.pop(0)
        if isinstance(naechste, Exception):
            raise naechste
        return naechste

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = ManagedLLMProvider(vendor="uni", model="testmodell", base_url="http://test/v1")
    sig = p.extract_signals("Ich arbeite an Information Retrieval.", "de")
    assert [s.value for s in sig] == ["information retrieval"]
    assert not folge   # zwei Fehlschlaege, dann Erfolg: alle drei Versuche verbraucht

def test_synthese_gibt_nach_drei_versuchen_auf(monkeypatch):
    p = _provider_mit_antworten(monkeypatch, [_transient()] * 3)
    with pytest.raises(RuntimeError, match="nach 3 Versuchen"):
        p.synthesize_text({"forschungsfelder": ["Information Retrieval"]})


def test_synthese_meldet_konfigurationsfehler_sofort(monkeypatch):
    """Nicht-transiente Fehler (Auth, Konfiguration) duerfen NICHT wiederholt werden."""
    fest = RuntimeError("uni-API: HTTP 401. Antwort: invalid key")
    p = _provider_mit_antworten(monkeypatch, [fest, "zweite Antwort waere falsch"])
    with pytest.raises(RuntimeError, match="401"):
        p.synthesize_text({"forschungsfelder": ["Information Retrieval"]})


def test_synthese_erkennt_rueckfrage_statt_profiltext():
    """Regression (gemma-4, Uni-API): fuenf von 300 Synthesen fragten nach den
    Daten, statt den Text zu schreiben. Die Antworten sind kurz und passierten die
    Laengengrenze; ohne diese Pruefung waeren sie als Profiltext ausgeliefert und
    eingebettet worden. Ein regulaerer Text darf dagegen nicht anschlagen."""
    rueckfragen = [
        "Da du das Profil noch nicht eingefügt hast, sende mir bitte die "
        "entsprechenden Daten zu.",
        "Es scheint, als ob der Text des Profils fehlt. Bitte füge die "
        "strukturierten Daten noch ein.",
        "Please provide the structured profile so that I can summarize it.",
    ]
    for t in rueckfragen:
        assert ManagedLLMProvider._ist_entartet(t) == "Rueckfrage statt Profiltext"
    regulaer = ("Als Postdoc liegen die Schwerpunkte in IT-Sicherheit und "
                "Kryptographie. Gesucht werden Kooperationen und Nachwuchs, "
                "den die Person betreuen kann.")
    assert ManagedLLMProvider._ist_entartet(regulaer) is None

def test_synthese_entartet_faellt_auf_kanonische_serialisierung_zurueck(monkeypatch):
    """Dokumentierter Rueckfall (Ausarbeitung, Plausibilitaetspruefung der Synthese).

    Bei einer entarteten Antwort liegt das strukturierte Profil vollstaendig vor;
    nur die sprachliche Fassung fehlt. Der Job darf deshalb nicht scheitern.
    """
    from provider import SyntheseEntartet

    class EntarteterProvider(MockProvider):
        def synthesize_text(self, structured):
            raise SyntheseEntartet("Synthese nach 3 Versuchen entartet: zu lang")

    miner = ProfileMiner(EntarteterProvider(), cfg={"representation": "text"})
    prof = miner.mine([Source(id="a.txt", is_own=True,
                              text="We study attention mechanisms and transformers "
                                   "for natural language processing.")])
    assert prof.text == _struct_to_text(prof.structured)
    assert prof.text and "fields:" in prof.text


def test_anbieterausfall_laesst_den_job_scheitern(monkeypatch):
    """Kein Rueckfall bei einem Anbieterausfall -- dort fehlen die Signale selbst."""
    class AusgefallenerProvider(MockProvider):
        def synthesize_text(self, structured):
            raise RuntimeError("Synthese nach 3 Versuchen fehlgeschlagen: HTTP 500")

    miner = ProfileMiner(AusgefallenerProvider(), cfg={"representation": "text"})
    with pytest.raises(RuntimeError, match="HTTP 500"):
        miner.mine([Source(id="a.txt", is_own=True,
                           text="We study attention mechanisms and transformers "
                                "for natural language processing.")])


# ---------- Antwortschema bei der Uni-API (Nachlauf zu Studie 1) ----------
def _uni_mit_post(monkeypatch, antwort, response_schema):
    """Uni-Provider, dessen _post die Nutzlasten mitschreibt und `antwort(payload)` liefert."""
    import provider as P
    p = ManagedLLMProvider(vendor="uni", model="testmodell", base_url="http://test/v1",
                           response_schema=response_schema)
    gesendet = []

    def fake_post(url, headers, payload):
        gesendet.append(payload)
        return antwort(payload)

    monkeypatch.setattr(p, "_post", fake_post)
    return p, gesendet, P


def _signal_antwort(_payload):
    import json
    return {"choices": [{"message": {"content": json.dumps({"signals": [
        {"kind": "field", "value": "Information Retrieval",
         "evidence": "arbeite an Information Retrieval", "confidence": 0.9}]})}}]}


def test_uni_fordert_schema_nur_im_signalschritt_an(monkeypatch):
    """Mit response_schema traegt der Signalschritt das JSON-Schema, die Synthese
    nicht -- sie soll Prosa liefern. Ohne die Option (Voreinstellung der Klasse,
    so liefen die Studien) geht kein response_format mit."""
    def antwort(payload):
        if "response_format" in payload:
            return _signal_antwort(payload)
        return {"choices": [{"message": {"content": "Eine Doktorandin arbeitet an Information Retrieval."}}]}

    p, gesendet, P = _uni_mit_post(monkeypatch, antwort, response_schema=True)
    sig = p.extract_signals("Ich arbeite an Information Retrieval.", "de")
    assert [s.value for s in sig] == ["information retrieval"]
    assert gesendet[0]["response_format"] == P.SIGNAL_RESPONSE_FORMAT
    p.synthesize_text({"forschungsfelder": ["Information Retrieval"]})
    assert "response_format" not in gesendet[-1]

    q, gesendet_q, _ = _uni_mit_post(monkeypatch, _signal_antwort, response_schema=False)
    q.extract_signals("Ich arbeite an Information Retrieval.", "de")
    assert "response_format" not in gesendet_q[0]


def test_uni_faellt_bei_abgelehntem_schema_zurueck(monkeypatch):
    """Lehnt der Dienst das Schema ab (HTTP 400), wiederholt der Provider den
    Aufruf ohne Schema und verzichtet danach dauerhaft darauf."""
    def antwort(payload):
        if "response_format" in payload:
            e = RuntimeError("uni-API: HTTP 400. Antwort: response_format not supported")
            e.status, e.transient = 400, False
            raise e
        return _signal_antwort(payload)

    p, gesendet, _ = _uni_mit_post(monkeypatch, antwort, response_schema=True)
    assert p.extract_signals("Ich arbeite an Information Retrieval.", "de")
    assert ["response_format" in g for g in gesendet] == [True, False]
    p.extract_signals("Ich arbeite an Information Retrieval.", "de")
    assert ["response_format" in g for g in gesendet] == [True, False, False]


def test_anderer_400_wird_nicht_als_schemaablehnung_gedeutet(monkeypatch):
    """Scheitert auch der Aufruf ohne Schema, lag der 400 nicht am Schema: Der
    Fehler wird gemeldet, und das Schema bleibt fuer spaetere Aufrufe aktiv."""
    def antwort(payload):
        e = RuntimeError("uni-API: HTTP 400. Antwort: unknown model")
        e.status, e.transient = 400, False
        raise e

    p, gesendet, _ = _uni_mit_post(monkeypatch, antwort, response_schema=True)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        p.extract_signals("Ich arbeite an Information Retrieval.", "de")
    assert p._schema_abgelehnt is False
