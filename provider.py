#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provider-Abstraktion fuer das Profilmining.

Zwei Operationen kapseln den austauschbaren Backend-Dienst:
  - extract_signals(chunk, lang) -> Liste belegter Signale  (Map-Schritt)
  - embed(text, dim, quant)      -> Vektor                  (Embedding)

MockProvider:        regelbasiert, laeuft OHNE API. Dient als sofort lauffaehige
                     Referenz UND als regelbasierte Vergleichsgroesse der Studien 2.2 und 4.
ManagedLLMProvider:  LLM-Zugriff ueber die Uni-LLM-API (OpenAI-kompatibel,
                     Basis-URL/Zugang via Deussen) oder direkte Modell-APIs
                     (OpenAI/Anthropic/Mistral) als Fallback bzw. Vergleichs-
                     modell im Zwei-LLM-Experiment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib, math, re


@dataclass
class Signal:
    kind: str          # 'field' | 'method' | 'role' | 'seek'
    value: str         # kanonischer Schluessel bzw. Wert
    evidence: str      # woertlicher Beleg (Satz/Span) -- Anti-Halluzination
    confidence: float  # 0..1


# ----------------------------------------------------------------------------------
# Kontrolliertes Vokabular (Surface-Form -> kanonischer Schluessel), DE + EN
# ----------------------------------------------------------------------------------
FIELD_VOCAB = {
    "maschinelles lernen": "ml", "machine learning": "ml", "supervised learning": "ml",
    "deep learning": "dl", "neuronale netze": "dl", "neural networks": "dl", "transformer": "dl",
    "sprachverarbeitung": "nlp", "natural language processing": "nlp", "nlp": "nlp", "sprachmodelle": "nlp", "language models": "nlp",
    "bildverarbeitung": "cv", "computer vision": "cv", "objekterkennung": "cv", "object detection": "cv",
    "bestaerkendes lernen": "rl", "reinforcement learning": "rl", "policy gradient": "rl",
    "information retrieval": "ir", "empfehlungssysteme": "ir", "recommender systems": "ir", "dense retrieval": "ir", "dichte retriever": "ir",
    "datenbanken": "db", "databases": "db", "anfrageoptimierung": "db", "query optimization": "db",
    "verteilte systeme": "dist", "distributed systems": "dist", "konsensprotokolle": "dist", "consensus": "dist",
    "it-sicherheit": "sec", "security": "sec", "datenschutz": "sec", "privacy": "sec", "kryptographie": "sec", "cryptography": "sec",
    "mensch-computer-interaktion": "hci", "human-computer interaction": "hci", "usability": "hci", "user studies": "hci", "nutzerstudien": "hci",
    "softwaretechnik": "se", "software engineering": "se", "continuous integration": "se", "design patterns": "se",
    "theoretische informatik": "theory", "complexity theory": "theory", "graphalgorithmen": "theory", "graph algorithms": "theory",
    "rechnernetze": "net", "computer networks": "net", "routing": "net", "mesh networking": "net", "drahtlose netze": "net",
    "robotik": "robo", "robotics": "robo", "motion planning": "robo", "bewegungsplanung": "robo", "slam": "robo",
    "bioinformatik": "bio", "bioinformatics": "bio", "sequenzanalyse": "bio", "genomics": "bio", "genomik": "bio",
    "computergrafik": "graph", "computer graphics": "graph", "rendering": "graph", "ray tracing": "graph", "raytracing": "graph",
}

# Methoden werden zusaetzlich woertlich als Signale gefuehrt (kanonisch = Kleinschreibung)
METHOD_HINTS = [
    "transformer", "backpropagation", "attention", "policy gradient", "q-learning",
    "differential privacy", "slam", "sensor fusion", "consensus", "sharding",
    "named entity recognition", "eigennamenerkennung", "ranking", "indexierung",
    "query optimization", "anfrageoptimierung", "rendering", "ray tracing",
    "sequence analysis", "sequenzanalyse", "graph algorithms", "graphalgorithmen",
]

ROLE_CUES = {
    "phd": ["doktorand", "phd student", "ph.d", "promovier", "dissertation"],
    "master": ["masterstudent", "master's student", "masterarbeit", "msc student"],
    "postdoc": ["postdoc", "post-doc", "postdoktorand"],
    "prof": ["professor", "faculty", "principal investigator", "ordinariat"],
    "industry": ["industrieforscher", "industry researcher", "research engineer", "im unternehmen", "at company"],
    "founder": ["gruender", "gründer", "founder", "start-up", "startup", "co-founder"],
}

SEEK_CUES = {
    "seek_mentor": ["suche mentor", "looking for mentor", "mentoring durch", "guidance from"],
    "offer_mentor": ["biete mentoring", "mentor junior", "begleite nachwuchs", "mentor students", "to mentor"],
    "seek_collab": ["kooperation", "collaborat", "zusammenarbeit", "joint work"],
    "seek_job": ["suche stelle", "looking for a position", "job in", "open to opportunities", "seeking a position"],
    "hiring": ["stelle zu besetzen", "hiring", "we are hiring", "talente fuer", "join my team", "join our team"],
    "seek_cofounder": ["mitgruender", "mitgründer", "co-founder", "cofounder"],
    "seek_feedback": ["feedback zu", "feedback on", "rueckmeldung zu"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class SyntheseEntartet(RuntimeError):
    """Die Synthese lieferte auch nach Wiederholung nur entartete Antworten.

    Eigener Typ, damit die Pipeline diesen Fall vom Anbieterausfall unterscheiden
    kann: Bei einer entarteten Antwort liegt das strukturierte Profil vollstaendig
    vor, es fehlt nur die sprachliche Fassung -- dann faellt der Dienst auf die
    kanonische Serialisierung zurueck (vgl. Ausarbeitung, Plausibilitaetspruefung
    der Synthese). Ein Anbieterausfall dagegen laesst den Job scheitern.
    """


class Provider(ABC):
    name = "abstract"
    # Erzeugt dieser Provider das Profil mit einem Sprachmodell? Daraus leitet der
    # Dienst die Kennzeichnung nach Art. 50 Abs. 2 KI-Verordnung ab (ai_generated).
    generative = True

    @abstractmethod
    def extract_signals(self, chunk: str, lang: str) -> list[Signal]:
        ...

    @abstractmethod
    def embed(self, text: str, dim: int = 256, quant: str = "float32"):
        ...


class MockProvider(Provider):
    """Regelbasiert, deterministisch, ohne API. Zugleich Referenz der Studien 2.2 und 4."""
    name = "mock-rule-based"
    generative = False

    def extract_signals(self, chunk: str, lang: str) -> list[Signal]:
        low = chunk.lower()
        out: list[Signal] = []
        sents = _sentences(chunk)

        def evidence_for(term):
            for s in sents:
                if term in s.lower():
                    return s[:240]
            return chunk[:240]

        seen = set()
        # Felder
        for surface, canon in FIELD_VOCAB.items():
            if surface in low and ("field", canon) not in seen:
                seen.add(("field", canon))
                out.append(Signal("field", canon, evidence_for(surface), 0.9))
        # Methoden (woertlich)
        for m in METHOD_HINTS:
            if m in low and ("method", m) not in seen:
                seen.add(("method", m))
                out.append(Signal("method", m, evidence_for(m), 0.7))
        # Rolle
        for canon, cues in ROLE_CUES.items():
            for cue in cues:
                if cue in low and ("role", canon) not in seen:
                    seen.add(("role", canon))
                    out.append(Signal("role", canon, evidence_for(cue), 0.8))
                    break
        # Suchabsicht
        for canon, cues in SEEK_CUES.items():
            for cue in cues:
                if cue in low and ("seek", canon) not in seen:
                    seen.add(("seek", canon))
                    out.append(Signal("seek", canon, evidence_for(cue), 0.75))
                    break
        return out

    NATIVE_DIM = 256   # Hash-Embedding hat keine Modellgroesse; das ist seine

    def embed(self, text: str, dim: int = 0, quant: str = "float32"):
        # Deterministisches Hash-Bag-of-Words-Embedding (Platzhalter fuer ein echtes
        # Modell) in fester Breite NATIVE_DIM. Kuerzung und Quantisierung laufen
        # danach ueber dieselben Funktionen wie beim ManagedLLMProvider.
        # Frueher bestimmte dim hier die Breite der Hash-Tabelle, und quant wurde
        # ignoriert: Mit der voreingestellten Referenz lieferte /embeddings trotz
        # quant="binary" Gleitkommazahlen. dim <= 0 bedeutet "native Groesse";
        # groessere Werte als NATIVE_DIM kuerzen nicht, sie verlaengern auch nicht.
        from embedding_compression import matryoshka_truncate, quantize
        vec = [0.0] * self.NATIVE_DIM
        toks = re.findall(r"[a-zA-ZäöüÄÖÜß]+", text.lower())
        for t in toks:
            if len(t) < 3:
                continue
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            vec[h % self.NATIVE_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = matryoshka_truncate([x / norm for x in vec], dim)
        vals, _ = quantize(vec, quant)
        return vals


# Prompt fuer den echten Managed-LLM-Map-Schritt (erzwungenes JSON).
EXTRACTION_PROMPT = """Du extrahierst belegte Interessen-Signale aus einem Textabschnitt
einer Person fuer das Matchmaking auf einer Konferenz.

Regeln:
- Stuetze JEDES Signal auf einen woertlichen Beleg aus dem Text (Feld "evidence").
- Erfinde nichts; wenn unsicher, lass es weg.
- kind in {field, method, role, seek}; confidence in [0,1].

Gib AUSSCHLIESSLICH valides JSON zurueck:
{"signals": [{"kind": "field", "value": "...", "evidence": "...", "confidence": 0.9}]}

Textabschnitt:
{chunk}
"""

SYNTHESIS_PROMPT = """Fasse das folgende strukturierte Konferenz-Profil zu einem
zusammenhaengenden Profiltext (3-4 Saetze) zusammen. Nenne Schwerpunkte, Methoden,
Rolle und gesuchte Kontakte; erfinde nichts, was nicht im Profil steht.
Antworte nur mit dem Text, ohne Einleitung.

Profil (JSON):
{profile}
"""


def extract_json_object(text: str) -> dict:
    """Toleranter Parser: entfernt Markdown-Zaeune/Vorspann, findet das erste
    balancierte JSON-Objekt in der Modellantwort."""
    import json
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.S).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("kein JSON-Objekt in der Antwort")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        elif ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unvollstaendiges JSON-Objekt in der Antwort")


# Antwortschema des Signalschritts. Bei der Uni-API (vLLM-Unterbau) wird es ueber
# response_format erzwungen, wenn der Provider mit response_schema=True laeuft.
# Hintergrund ist der Nachlauf zu Studie 1: llama-3.1-8b lieferte ohne Schema in
# einem Teil der Faelle statt Daten einen nachgeahmten Werkzeugaufruf oder
# Python-Code, mit Schema in 30 von 30 Faellen verwertbares JSON.
SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["field", "method", "role", "seek"]},
                    "value": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "value", "evidence", "confidence"],
            },
        }
    },
    "required": ["signals"],
}
SIGNAL_RESPONSE_FORMAT = {"type": "json_schema",
                          "json_schema": {"name": "signale", "schema": SIGNAL_SCHEMA}}


class ManagedLLMProvider(Provider):
    """LLM-Zugriff ueber eine konfigurierbare Modell-API. Primaerer Weg ist die
    kostenfreie Uni-LLM-API (vendor="uni", OpenAI-kompatibles Chat-Completions-
    Schema; Basis-URL ueber LLM_BASE_URL, Zugangsdaten via Deussen). Direkte
    Anbieter-APIs (OpenAI/Anthropic/Mistral) bleiben als Vergleichsmodelle fuer
    das Zwei-LLM-Experiment nutzbar. Verwendet die Standard-Bibliothek (urllib),
    keine SDK-Abhaengigkeit.

    Das Embedding laeuft bewusst NICHT ueber die LLM-API, sondern lokal ueber
    ein quelloffenes sentence-transformers-Modell -- dieselben Modelle wie in
    den Studien 3.1 und 3.2, on-device-faehig und kostenfrei.

    Produktivbetrieb Conference-App: Derselbe Provider-Vertrag laesst sich auf
    einen Hyperscaler in EU-Region (Vertex AI / Bedrock / Azure OpenAI) legen;
    es aendern sich nur Endpoint-URL, Authentifizierung (Service-Account statt
    API-Key) und Modellname. Diese Anbindung ist Teil des Dissertationsprojekts.
    """
    name = "managed-llm"

    ENV_KEYS = {"uni": "UNI_LLM_API_KEY", "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", "mistral": "MISTRAL_API_KEY"}
    DEFAULT_MODELS = {"uni": "", "openai": "gpt-4o-mini",
                      "anthropic": "claude-haiku-4-5", "mistral": "mistral-small-latest"}

    def __init__(self, vendor="uni", model="", embed_model="", temperature=0.3,
                 timeout=60, base_url="", prompt_template="", require_evidence=True,
                 verify_evidence=False, response_schema=False):
        """prompt_template / require_evidence / verify_evidence steuern die
        Prompt-Studie (Studie 2.1). Voreinstellung = Produktivverhalten.

        require_evidence : Signale ohne Belegfeld verwerfen (Schema-Ebene).
        verify_evidence  : zusaetzlich pruefen, ob der Beleg WOERTLICH im Chunk
                           steht. Bis dahin war nur "Feld nicht leer" gefordert --
                           ein Modell konnte Wert und Beleg gemeinsam erfinden.
        response_schema  : bei der Uni-API das Antwortschema des Signalschritts
                           anfordern (SIGNAL_RESPONSE_FORMAT). Die Dienst-
                           dokumentation sichert das nicht zu; lehnt der Dienst
                           ab (HTTP 400), faellt der Provider einmalig und dann
                           dauerhaft auf den freien Modus zurueck. Die Klasse
                           laesst es aus, damit die Studien reproduzierbar bleiben
                           (sie liefen ohne Schema); der Dienst schaltet es ein.
        """
        import os
        if vendor not in self.ENV_KEYS:
            raise ValueError(f"vendor muss in {sorted(self.ENV_KEYS)} liegen, nicht '{vendor}'")
        self.vendor = vendor
        # Uni-API: Basis-URL und Modellname kommen von Deussen (LLM_BASE_URL/LLM_MODEL);
        # gegen die OpenAI-kompatible Chat-Completions-Route implementiert.
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        # Modellwahl je Anbieter getrennt halten. LLM_MODEL gilt AUSSCHLIESSLICH
        # fuer die Uni-API, weil dort kein sinnvoller Default existiert. Frueher
        # galt die Variable fuer alle Anbieter -- wer sie fuer die Uni-API gesetzt
        # hatte und danach auf Mistral wechselte, schickte den Uni-Modellnamen an
        # Mistral und bekam HTTP 400 "Invalid model". Fuer die uebrigen Anbieter
        # gibt es eigene, optionale Variablen.
        if model:
            self.model = model
        elif vendor == "uni":
            self.model = os.environ.get("LLM_MODEL", "")
        else:
            self.model = (os.environ.get(f"{vendor.upper()}_MODEL")
                          or self.DEFAULT_MODELS[vendor])
            if os.environ.get("LLM_MODEL"):
                print(f"  Hinweis: LLM_MODEL={os.environ['LLM_MODEL']!r} ist gesetzt, gilt aber "
                      f"nur fuer die Uni-API. {vendor} nutzt {self.model!r} "
                      f"(ueberschreibbar mit {vendor.upper()}_MODEL).")
        self.embed_model = embed_model or os.environ.get(
            "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.temperature, self.timeout = temperature, timeout
        self.prompt_template = prompt_template or EXTRACTION_PROMPT
        self.require_evidence = require_evidence
        self.verify_evidence = verify_evidence
        self.response_schema = response_schema
        self._schema_abgelehnt = False
        self.last_evidence_stats = {"signale": 0, "beleg_woertlich": 0, "beleg_fehlt": 0}
        if vendor == "uni" and timeout == 60:
            # Geteilter Uni-Dienst: unter Last dauern Antworten laenger als bei
            # kommerziellen APIs; grosszuegigeres Timeout statt Abbruch.
            self.timeout = 120
        self.api_key = os.environ.get(self.ENV_KEYS[vendor], "")
        self._st_model = None  # lazy geladenes Embedding-Modell
        import threading
        self._model_lock = threading.Lock()   # schuetzt das einmalige Laden

    # ---- HTTP (Standard-Bibliothek, anbieterspezifische Schemata) ----
    # --- Drosselung -----------------------------------------------------------
    # Geteilte Hochschuldienste begrenzen die Anfragerate. Ein Lauf ueber den
    # Dokumentkorpus setzt rund 2100 Aufrufe ab und laeuft ohne Bremse in HTTP 429.
    # Der Mindestabstand gilt prozessweit (Klassenattribut), weil alle Aufrufe
    # denselben Dienst treffen, und er passt sich selbst an: Jede 429-Antwort
    # erhoeht ihn, sodass der Lauf sich ohne Vorwissen ueber das Limit einpendelt.
    _last_request = 0.0
    _min_interval = float(__import__("os").environ.get("LLM_MIN_INTERVAL", "0") or 0)
    _MAX_INTERVAL = 30.0
    # Der Mindestabstand wird unter dieser Sperre reserviert. Ohne sie las jeder
    # Thread denselben Wert von _last_request, berechnete eine Wartezeit von 0 und
    # feuerte gleichzeitig: bei zehn parallelen Aufrufen und einem Mindestabstand
    # von 1 s lagen acht von neun Abstaenden bei 0,000 s. Die Drosselung bremste
    # damit den Einzelfall, verhinderte den HTTP 429 unter Last aber nicht.
    _throttle_lock = __import__("threading").Lock()

    @classmethod
    def _throttle(cls):
        import time
        if cls._min_interval <= 0:
            return
        with cls._throttle_lock:
            # Naechsten Startzeitpunkt reservieren, BEVOR die Sperre fallen
            # gelassen wird -- sonst ist der reservierte Platz wieder frei.
            jetzt = time.time()
            start = max(jetzt, cls._last_request + cls._min_interval)
            cls._last_request = start
        wartet = start - jetzt
        if wartet > 0:
            time.sleep(wartet)

    @classmethod
    def _drosseln_nach_429(cls, retry_after: float | None):
        """Mindestabstand erhoehen, nachdem der Dienst abgewiesen hat."""
        with cls._throttle_lock:
            vorschlag = retry_after if retry_after else max(1.0, cls._min_interval) * 1.5
            cls._min_interval = min(cls._MAX_INTERVAL, max(cls._min_interval + 0.5, vorschlag))
        print(f"  -> Anfragerate gedrosselt: Mindestabstand jetzt "
              f"{cls._min_interval:.1f}s zwischen Aufrufen")

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        import json, time, urllib.request, urllib.error
        self._throttle()
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                antwort = json.loads(r.read().decode("utf-8"))
            # Kein Nachsetzen von _last_request: _throttle() hat den Startzeitpunkt
            # bereits reserviert. Ein Ueberschreiben mit "jetzt" wuerde die
            # Reservierung nachfolgender Threads zurueckdatieren.
            return antwort
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                detail = ""
            hint = ""
            if e.code == 429:
                try:
                    retry_after = float(e.headers.get("Retry-After") or 0) or None
                except (TypeError, ValueError):
                    retry_after = None
                type(self)._drosseln_nach_429(retry_after)
                if "quota" in detail.lower() or "billing" in detail.lower():
                    hint = " (Hinweis: 429 kann auch fehlendes Guthaben bedeuten)"
            err = RuntimeError(
                f"{self.vendor}-API: HTTP {e.code}{hint}. Antwort: {detail}")
            err.status = e.code
            err.transient = e.code >= 500 or e.code == 429   # Server-/Lastfehler: Retry sinnvoll
            err.rate_limited = e.code == 429
            raise err from None
        except urllib.error.URLError as e:
            # Verbindungsfehler gelten als voruebergehend. Die Wiederholung in
            # extract_signals/synthesize_text greift nur bei e.transient; frueher
            # fehlte die Markierung hier, und schon ein einzelner Verbindungs-
            # abbruch liess den Job scheitern, obwohl der Kommentar das Gegenteil
            # versprach.
            err = RuntimeError(f"{self.vendor}-API nicht erreichbar: {e.reason}")
            err.transient = True
            raise err from None
        except TimeoutError:
            # Socket-Timeout mitten in der Antwort (z.B. ausgelasteter Uni-Dienst):
            # ebenfalls voruebergehend, also wiederholen statt abbrechen.
            err = RuntimeError(
                f"{self.vendor}-API: Timeout nach {self.timeout}s (Dienst ausgelastet?)")
            err.transient = True
            raise err from None

    def _chat(self, prompt: str, response_format: dict | None = None) -> str:
        # Uni-API laeuft ggf. netzintern ohne Key; alle anderen Anbieter brauchen einen.
        if not self.api_key and self.vendor != "uni":
            raise RuntimeError(f"{self.ENV_KEYS[self.vendor]} ist nicht gesetzt")
        if self.vendor == "uni":
            if not self.base_url:
                raise RuntimeError(
                    "LLM_BASE_URL ist nicht gesetzt (Basis-URL der Uni-LLM-API, "
                    "Zugangsinfos via Deussen)")
            if not self.model:
                raise RuntimeError("LLM_MODEL ist nicht gesetzt (Modellname der Uni-LLM-API)")
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            # Kein response_format: JSON-Modus ist bei fremd gehosteten Modellen nicht
            # garantiert; extract_json_object parst die Antwort tolerant.
            payload = {"model": self.model, "temperature": self.temperature,
                       # explizit setzen: der Uni-Dienst kappt sonst bei einem knappen
                       # Default und liefert abgeschnittenes (unparsbares) JSON
                       "max_tokens": 2000,
                       "messages": [{"role": "user", "content": prompt}]}
            if response_format is not None:
                payload["response_format"] = response_format
            data = self._post(
                self.base_url.rstrip("/") + "/chat/completions", headers, payload)
            u = data.get("usage", {})
            self.last_usage = {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}
            return data["choices"][0]["message"]["content"]
        if self.vendor == "anthropic":
            data = self._post(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                {"model": self.model, "max_tokens": 1500, "temperature": self.temperature,
                 "messages": [{"role": "user", "content": prompt}]})
            u = data.get("usage", {})
            self.last_usage = {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}
            return data["content"][0]["text"]
        base = {"openai": "https://api.openai.com/v1/chat/completions",
                "mistral": "https://api.mistral.ai/v1/chat/completions"}[self.vendor]
        data = self._post(
            base, {"Authorization": f"Bearer {self.api_key}"},
            {"model": self.model, "temperature": self.temperature,
             "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": prompt}]})
        u = data.get("usage", {})
        self.last_usage = {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}
        return data["choices"][0]["message"]["content"]

    def _signal_anfrage(self, prompt: str) -> str:
        """Ein Aufruf des Signalschritts; bei der Uni-API auf Wunsch mit Schema.

        Lehnt der Dienst das Schema mit HTTP 400 ab, wird derselbe Aufruf ohne
        Schema wiederholt. Gelingt er so, lag es am Schema, und der Provider
        verzichtet fuer den Rest des Prozesses darauf. Scheitert auch der freie
        Aufruf, hat der 400 einen anderen Grund und wird weitergereicht.
        """
        if not (self.vendor == "uni" and self.response_schema and not self._schema_abgelehnt):
            return self._chat(prompt)
        try:
            return self._chat(prompt, response_format=SIGNAL_RESPONSE_FORMAT)
        except RuntimeError as e:
            if getattr(e, "status", None) != 400:
                raise
            raw = self._chat(prompt)
            self._schema_abgelehnt = True
            import logging
            logging.getLogger("profile_mining").warning(
                "Uni-API lehnt das Antwortschema ab (HTTP 400, Modell %s); "
                "Signalschritt laeuft ab jetzt ohne Schema.", self.model)
            return raw

    # ---- Provider-Vertrag ----
    EXTRACT_ATTEMPTS = 3   # unparsbare Antworten (z.B. abgeschnittenes JSON ohne
                           # erzwungenen JSON-Modus, vgl. Studie 1) direkt neu anfragen

    def extract_signals(self, chunk: str, lang: str) -> list[Signal]:
        import time
        obj, raw, last_err = None, "", None
        for attempt in range(self.EXTRACT_ATTEMPTS):
            try:
                raw = self._signal_anfrage(self.prompt_template.replace("{chunk}", chunk))
            except RuntimeError as e:
                # Transiente Anbieterfehler (HTTP 5xx/429) mit kurzem Backoff
                # wiederholen; alles andere (Konfiguration, Auth) sofort melden.
                if not getattr(e, "transient", False):
                    raise
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            try:
                obj = extract_json_object(raw)
                break
            except ValueError:
                # Modell hat kein (vollstaendiges) JSON geliefert -- leere Antwort,
                # Floskel, Reasoning-Vorspann oder mitten im Objekt abgeschnitten.
                # Naechster Versuch; nach dem letzten Fehlschlag Diagnose ausgeben.
                continue
        if obj is None:
            if last_err is not None and not raw:
                raise RuntimeError(
                    f"nach {self.EXTRACT_ATTEMPTS} Versuchen: {last_err}") from None
            raise RuntimeError(
                f"{self.vendor}/{self.model}: nach {self.EXTRACT_ATTEMPTS} Versuchen "
                f"keine Antwort mit JSON-Objekt: {raw[:200]!r}")
        out = []
        for s in obj.get("signals", []):
            kind, value = s.get("kind"), str(s.get("value", "")).strip()
            evidence = str(s.get("evidence", "")).strip()
            if kind not in ("field", "method", "role", "seek") or not value:
                continue
            st = self.last_evidence_stats
            st["signale"] += 1
            if not evidence:
                st["beleg_fehlt"] += 1
                if self.require_evidence:
                    continue
            else:
                # Steht der Beleg woertlich im Abschnitt? Normalisiert auf
                # Kleinschreibung und Leerraum, damit reine Formatierungs-
                # unterschiede nicht als Erfindung zaehlen.
                norm = lambda t: " ".join(t.lower().split())
                woertlich = norm(evidence) in norm(chunk)
                if woertlich:
                    st["beleg_woertlich"] += 1
                elif self.verify_evidence:
                    continue   # Beleg steht nicht im Text -> verwerfen
            try:
                conf = max(0.0, min(1.0, float(s.get("confidence", 0.5))))
            except (TypeError, ValueError):
                conf = 0.5
            # kanonische Form: Werte kleingeschrieben (vgl. Vokabular/Tests)
            out.append(Signal(kind=kind, value=value.lower(),
                              evidence=evidence, confidence=conf))
        return out

    @staticmethod
    def _unwrap_prose(raw: str) -> str:
        """Entpackt Prosa, die das Modell in ein JSON-Objekt verpackt hat.

        Hintergrund: _chat() setzt fuer OpenAI/Mistral response_format=json_object,
        weil die Signalextraktion striktes JSON braucht. Der Synthese-Prompt
        verlangt dagegen reine Prosa -- unter erzwungenem JSON-Modus verpackt das
        Modell den Absatz dann in ein Objekt mit frei gewaehltem Schluessel
        ("profiltext", "text", "zusammenfassung", ...). Ohne dieses Entpacken
        landet die JSON-Huelle im Profiltext und damit im Embedding.
        Antworten, die kein JSON-Objekt sind (z.B. von der Uni-API, die keinen
        JSON-Modus erzwingt), bleiben unveraendert.
        """
        t = (raw or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.S).strip()
        if not t.startswith("{"):
            return t
        try:
            obj = extract_json_object(t)
        except (ValueError, Exception):  # noqa: B014 - tolerant, Rohtext als Fallback
            return t
        if not isinstance(obj, dict):
            return t
        # laengster String-Wert = der eigentliche Profiltext, unabhaengig vom Key
        vals = [v.strip() for v in obj.values() if isinstance(v, str) and v.strip()]
        return max(vals, key=len) if vals else t

    # Ein Profiltext sind laut Prompt 3-4 Saetze. Alles jenseits dieser Grenze ist
    # keine lange Antwort, sondern eine entgleiste.
    MAX_PROFILTEXT_ZEICHEN = 1500

    # Rueckfragen und Verweigerungen statt eines Profiltexts. Anlass: Im gemma-4-Lauf
    # ueber die Uni-API antworteten fuenf von 300 Synthesen sinngemaess "Da du das
    # Profil noch nicht eingefuegt hast, ..." (p0006, p0034, p0103, p0105, p0268).
    # Solche Antworten sind kurz und passieren die Laengengrenze. Die Muster sind eng
    # gefasst: Ein Profiltext spricht den Leser nie an. Ueber alle 9690 vorhandenen
    # Profiltexte (Template, Mistral, gemma-4, Prompt- und Upload-Varianten) schlagen
    # sie genau bei diesen fuenf an.
    _RUECKFRAGE = re.compile(
        r"(nicht eingef[uü]gt|bitte (poste|sende|schick|f[uü]ge|gib)|sobald du|"
        r"please (provide|paste|share|send)|once you (provide|share|paste)|"
        r"i (can't|cannot|am unable to) (write|create|summari[sz]e))",
        re.IGNORECASE)

    @staticmethod
    def _ist_entartet(text: str) -> str | None:
        """Erkennt Wiederholungsschleifen, uebermaessige Laenge und Rueckfragen.

        Anlass: Bei einem Lauf ueber die Uni-API entartete eine von 300 Antworten
        in eine Schleife; fuenf weitere waren Rueckfragen (siehe _RUECKFRAGE) ("... own own own own ...", rund 8000 Zeichen). Anbieter
        mit erzwungenem JSON-Modus bremsen solche Faelle implizit ab; wo dieser
        Modus fehlt, muss der Client selbst pruefen. Rueckgabe: Grund oder None.
        """
        t = (text or "").strip()
        if ManagedLLMProvider._RUECKFRAGE.search(t):
            return "Rueckfrage statt Profiltext"
        if len(t) > ManagedLLMProvider.MAX_PROFILTEXT_ZEICHEN:
            return f"zu lang ({len(t)} Zeichen, erlaubt {ManagedLLMProvider.MAX_PROFILTEXT_ZEICHEN})"
        woerter = t.split()
        if len(woerter) >= 12:
            # gleiches Wort >= 8 mal hintereinander
            lauf, letztes = 1, None
            for w in woerter:
                lauf = lauf + 1 if w == letztes else 1
                letztes = w
                if lauf >= 8:
                    return f"Wiederholungsschleife (ein Wort {lauf}x hintereinander)"  # Wort selbst nicht ins Log (NFA2)
            # sehr geringe Wortvielfalt trotz Laenge
            if len(set(woerter)) / len(woerter) < 0.15:
                return f"kaum Wortvielfalt ({len(set(woerter))} verschiedene bei {len(woerter)} Woertern)"
        return None

    def synthesize_text(self, structured: dict) -> str:
        """LLM-Synthese des Profiltexts (Stufe 6): das Sprachmodell formuliert den
        zusammenhaengenden Absatz aus dem strukturierten Profil.

        Wiederholt wie extract_signals: transiente Anbieterfehler (HTTP 5xx/429)
        und entartete Antworten fuehren zu einem neuen Versuch, statt den ganzen
        Job zu verwerfen. Frueher stand hier ein einzelner Aufruf mit dem Hinweis,
        die "Wiederholungslogik des Aufrufers" greife -- die gab es an dieser
        Stelle aber nicht (pipeline.mine ruft die Synthese genau einmal auf). Ein
        einzelner 500er im letzten Schritt verwarf damit die gesamte vorher
        geleistete Extraktion samt ihrer Kosten.
        """
        import json, time
        prompt = SYNTHESIS_PROMPT.replace(
            "{profile}", json.dumps(structured, ensure_ascii=False))
        last_err, letzte_war_entartet = None, False
        for attempt in range(self.EXTRACT_ATTEMPTS):
            try:
                raw = self._chat(prompt)
            except RuntimeError as e:
                if not getattr(e, "transient", False):
                    raise
                last_err, letzte_war_entartet = e, False
                time.sleep(1.5 * (attempt + 1))
                continue
            text = self._unwrap_prose(raw)
            grund = self._ist_entartet(text)
            if not grund:
                return text
            last_err = RuntimeError(f"Synthese-Antwort verworfen: {grund}")
            letzte_war_entartet = True
        if letzte_war_entartet:
            raise SyntheseEntartet(
                f"Synthese nach {self.EXTRACT_ATTEMPTS} Versuchen entartet: {last_err}")
        raise RuntimeError(
            f"Synthese nach {self.EXTRACT_ATTEMPTS} Versuchen fehlgeschlagen: {last_err}")

    # Praefixe der instruktionsgetunten Embedding-Modelle. Identisch zu MODELS in
    # run_experiments_local.py -- beide Wege muessen dieselben Vektoren erzeugen,
    # sonst misst die Evaluation etwas anderes als der Dienst ausliefert.
    EMBED_PREFIXES = {
        "intfloat/multilingual-e5-large": "query: ",
        "intfloat/multilingual-e5-base": "query: ",
        "intfloat/multilingual-e5-small": "query: ",
        "Snowflake/snowflake-arctic-embed-l-v2.0": "query: ",
    }

    def _embed_prefix(self) -> str:
        return self.EMBED_PREFIXES.get(self.embed_model, "")

    def embed(self, text: str, dim: int = 0, quant: str = "float32"):
        # Embedding laeuft bewusst LOKAL ueber ein quelloffenes sentence-transformers-
        # Modell (identisch zu den Studien 3.1/3.2, on-device-faehig) -- unabhaengig vom
        # LLM-Anbieter und kostenfrei.
        from embedding_compression import matryoshka_truncate, quantize
        if self._st_model is None:
            # Unter der Sperre erneut pruefen: der Dienst teilt eine Provider-
            # Instanz ueber alle Anfragen (service.make_provider). Ohne diese
            # Pruefung wuerden zwei gleichzeitig eintreffende Anfragen das Modell
            # doppelt laden -- genau die Speicherverschwendung, die das Teilen
            # der Instanz vermeiden soll.
            with self._model_lock:
                if self._st_model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError:
                        raise RuntimeError(
                            "sentence-transformers nicht installiert "
                            "(pip install sentence-transformers torch)") from None
                    self._st_model = SentenceTransformer(self.embed_model)
        # Instruktionsgetunte Modelle erwarten ein Praefix; ohne dieses liefern sie
        # messbar schwaechere Vektoren. Der Auswertungspfad der Studien 3.1/3.2 setzt es
        # (run_experiments_local.py, MODELS), der Dienst tat es bisher nicht -- bei
        # einem Wechsel des Embedding-Modells waeren die ausgelieferten Vektoren
        # damit schlechter gewesen als die gemessenen, ohne sichtbaren Fehler.
        vec = self._st_model.encode([self._embed_prefix() + text],
                                    normalize_embeddings=True)[0].tolist()
        vec = matryoshka_truncate(vec, dim)
        vals, _ = quantize(vec, quant)
        return vals
