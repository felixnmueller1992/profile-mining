#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profilmining-Pipeline (Map-Reduce mit Belegbindung).

Stufen:
  1 Ingestion/Normalisierung  2 Chunking (mit Herkunft)  3 Map (Provider, belegt)
  4 Reduce (Gruppierung + Salienz)  5 Verifikation (Beleg + Salienz-Schwelle)
  6 Synthese (strukturiert + Freitext)  7 Embedding (+ optional Kompression)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import re

from provider import Provider, Signal, SyntheseEntartet

log = logging.getLogger("profile_mining")

AREA_LABEL = {
    "ml": "Machine Learning", "dl": "Deep Learning", "nlp": "Natural Language Processing",
    "cv": "Computer Vision", "rl": "Reinforcement Learning", "ir": "Information Retrieval",
    "db": "Databases", "dist": "Distributed Systems", "sec": "Security & Privacy",
    "hci": "Human-Computer Interaction", "se": "Software Engineering", "theory": "Theory & Algorithms",
    "net": "Computer Networks", "robo": "Robotics", "bio": "Bioinformatics", "graph": "Computer Graphics",
}
ROLE_LABEL = {"phd": "PhD student", "master": "Master's student", "postdoc": "Postdoc",
              "prof": "Professor", "industry": "Industry researcher", "founder": "Startup founder"}
SEEK_LABEL = {"seek_mentor": "mentorship", "offer_mentor": "to mentor juniors",
              "seek_collab": "research collaborations", "seek_job": "a position",
              "hiring": "to hire", "seek_cofounder": "a co-founder", "seek_feedback": "feedback"}


@dataclass
class Source:
    id: str
    text: str
    lang: str = "auto"
    is_own: bool = True   # eigene Arbeit (hohe Autoritaet) vs. nur gelesen


@dataclass
class Profile:
    structured: dict
    text: str
    signals: list                       # aggregierte Signale mit Salienz + Beleg
    embedding: list                     # ein Vektor fuer das Gesamtprofil
    term_embeddings: list | None = None # optional: je Profilterm ein Vektor (Prof-Vorschlag)


def detect_lang(text):
    de = len(re.findall(r"[äöüßÄÖÜ]|\b(und|der|die|ich|mit|für|auf)\b", text.lower()))
    en = len(re.findall(r"\b(the|and|with|for|research|looking|i am)\b", text.lower()))
    return "de" if de >= en else "en"


def chunk_text(text, max_chars=1200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + 200:
                end = cut + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if len(c) >= 30]


class MiningAbgebrochen(Exception):
    """Der Auftraggeber hat den Lauf abgebrochen, etwa weil der Job geloescht wurde."""


class ProfileMiner:
    def __init__(self, provider: Provider, cfg: dict | None = None):
        self.p = provider
        # authority_read 0.8: Wer ein fremdes Dokument hochlaedt, tut das in einer
        # Matching-App absichtlich -- der Upload IST die Interessensbekundung,
        # unabhaengig davon, wer den Text verfasst hat. Das Thema soll deshalb nur
        # leicht hinter den selbst bearbeiteten liegen, nicht halb so schwer wiegen.
        #
        # Der frueher gesetzte Wert 0.5 hatte zwei unerwuenschte Nebenwirkungen.
        # Erstens ist die Autoritaet relativ gemeint, min_salience aber absolut:
        # Bei einer einzigen gelesenen Quelle musste die Konfidenz ueber 0.8 liegen
        # (0.4 / 0.5), sonst fiel das Signal heraus. Wer NUR fremde Arbeiten
        # hochlaedt -- Studierende, Praktiker ohne eigene Publikationen -- verlor
        # dadurch alles, wovon das Modell nicht fast voellig ueberzeugt war. Mit 0.8
        # sinkt die noetige Konfidenz auf 0.5. Zweitens wurde ein aelteres eigenes
        # Dokument gegenueber neuen, gelesenen Interessen stark bevorzugt.
        #
        # Die Rangfolge eigen vor gelesen sichert die Autoritaet. Der
        # Mehrquellen-Bonus vergroessert nur den Abstand zum Primaerthema, das
        # typisch in mehreren Quellen steht (Paper, Lebenslauf, Webseite); gegen
        # ein Sekundaerthema, das wie das gelesene nur in einer Quelle steht,
        # hilft er nicht. Die Ablation in Studie 2.2 zeigt: Bei Schwelle 0.4
        # bleiben Feld-F1, Recall, Precision, Rangfolge und Fremdrollenrate ueber
        # 0.5 bis 0.9 unveraendert, auch ohne Mehrquellen-Bonus. Bei 1.0 stehen
        # fremde und eigene Rolle gleichauf, und die Upload-Reihenfolge
        # entscheidet (auswertung_studie2.py).
        self.cfg = {"max_chars": 1200, "min_salience": 0.4, "authority_own": 1.0,
                    "authority_read": 0.8,
                    # 0 = native Dimension des Modells, also KEINE Kuerzung.
                    # FA6: Der Dienst liefert den vollen Vektor, die Kompression
                    # fuer den Mesh-Austausch entscheidet das Endgeraet ueber die
                    # Parameter dim/quant des Embedding-Endpunkts.
                    "embed_dim": 0, "representation": "text",
                    # beta aus Gleichung (Salienz): Zuschlag je zusaetzlicher Quelle,
                    # in der dasselbe Signal auftritt. Stand bis zuletzt als Literal
                    # in der Formel und war damit -- entgegen der Beschreibung im
                    # Implementierungskapitel -- nicht ablatierbar.
                    "multi_source_bonus": 0.3,
                    "verify": True,
                    # Prof-Vorschlag (Mail 07/2026): statt (oder neben) einem
                    # Gesamtprofil-Vektor je Profilterm einen Vektor zurueckgeben.
                    # Tradeoff Mesh-Payload vs. erklaerbares Matching -> FF3-Experiment.
                    "term_embeddings": False, "max_terms": 8,
                    # Kappung der Ergebnislisten. Standen bis zuletzt als Literale
                    # in der Synthese und waren damit -- wie zuvor schon der
                    # Mehrquellen-Bonus -- nicht ablatierbar. Das faellt erst auf,
                    # wenn ein Extraktor genug Signale liefert, dass die Grenze
                    # ueberhaupt bindet: Auf dem Dokumentkorpus tut die
                    # regelbasierte Referenz das nie, ein Sprachmodell in jedem
                    # Profil. 0 bedeutet "nicht kappen".
                    "max_fields": 5, "max_methods": 6, "max_seeks": 3,
                    **(cfg or {})}

    def mine(self, sources: list[Source], abbrechen=None) -> Profile:
        """abbrechen: optionale Funktion ohne Argumente. Liefert sie True, endet
        der Lauf vor dem naechsten Modellaufruf mit MiningAbgebrochen. Sonst
        rechnete ein geloeschter Job alle Chunks zu Ende und verbraucht
        Anbieterkontingent fuer ein Ergebnis, das niemand mehr abholt."""
        def pruefen():
            if abbrechen is not None and abbrechen():
                raise MiningAbgebrochen()

        # 1-2 Ingestion + Chunking mit Herkunft
        chunks = []  # (source, chunk_text)
        for s in sources:
            if len(s.text.strip()) < 30:
                continue
            lang = detect_lang(s.text) if s.lang == "auto" else s.lang
            for c in chunk_text(s.text, self.cfg["max_chars"]):
                chunks.append((s, lang, c))

        # 3 Map: belegte Signale je Chunk
        raw = []  # (Signal, Source)
        for s, lang, c in chunks:
            pruefen()                       # zwischen zwei Chunks, je ein Modellaufruf
            for sig in self.p.extract_signals(c, lang):
                raw.append((sig, s))

        # 4 Reduce: gruppieren + Salienz
        groups = defaultdict(list)  # (kind,value) -> list[(Signal,Source)]
        for sig, s in raw:
            groups[(sig.kind, sig.value)].append((sig, s))

        agg = []
        for (kind, value), occ in groups.items():
            n_sources = len({s.id for _, s in occ})
            authority = max((self.cfg["authority_own"] if s.is_own else self.cfg["authority_read"])
                            for _, s in occ)
            avg_conf = sum(sig.confidence for sig, _ in occ) / len(occ)
            beta = self.cfg["multi_source_bonus"]
            salience = round(avg_conf * authority * (1 + beta * (n_sources - 1)), 3)
            best = max(occ, key=lambda x: x[0].confidence)[0]
            agg.append({"kind": kind, "value": value, "salience": salience,
                        "n_sources": n_sources, "evidence": best.evidence})

        # 5 Verifikation: ohne Beleg / unter Schwelle -> verwerfen
        if self.cfg["verify"]:
            agg = [a for a in agg if a["evidence"] and a["salience"] >= self.cfg["min_salience"]]
        agg.sort(key=lambda a: a["salience"], reverse=True)

        # 6 Synthese beider Repraesentationen aus DERSELBEN Aggregations-Schicht
        fields = [a for a in agg if a["kind"] == "field"]
        methods = [a for a in agg if a["kind"] == "method"]
        roles = [a for a in agg if a["kind"] == "role"]
        seeks = [a for a in agg if a["kind"] == "seek"]

        def kappen(liste, schluessel):
            n = self.cfg[schluessel]
            return liste if not n else liste[:n]

        structured = {
            "forschungsfelder": [AREA_LABEL.get(a["value"], a["value"])
                                 for a in kappen(fields, "max_fields")],
            "methoden": [a["value"] for a in kappen(methods, "max_methods")],
            "rolle": ROLE_LABEL.get(roles[0]["value"], roles[0]["value"]) if roles else None,
            "sucht": [SEEK_LABEL.get(a["value"], a["value"])
                      for a in kappen(seeks, "max_seeks")],
        }
        # Im Managed-Betrieb schreibt das LLM den Profiltext (synthesize_text),
        # die regelbasierte Referenz nutzt die deterministische Synthese.
        synth = getattr(self.p, "synthesize_text", None)
        pruefen()                           # vor der Synthese, dem letzten Modellaufruf
        if callable(synth):
            try:
                text = synth(structured)
            except SyntheseEntartet as e:
                # Dokumentierter Rueckfall: Das strukturierte Profil liegt
                # vollstaendig vor, nur die sprachliche Fassung ist unbrauchbar.
                # Statt den Job zu verwerfen, tritt die kanonische Serialisierung
                # an ihre Stelle. Ein Anbieterausfall wird NICHT abgefangen -- dort
                # fehlen die Signale selbst, und der Job scheitert.
                log.warning("Synthese entartet, kanonische Serialisierung "
                            "ersatzweise verwendet: %s", e)
                text = _struct_to_text(structured)
        else:
            text = self._synthesize_text(structured)

        # 7 Embedding der gewaehlten Repraesentation
        rep = text if self.cfg["representation"] == "text" else _struct_to_text(structured)
        emb = self.p.embed(rep, dim=self.cfg["embed_dim"])

        # 7b optional: term-weise Embeddings (ein Vektor je saliente(m) Term/Signal)
        term_embs = None
        if self.cfg["term_embeddings"]:
            term_embs = [{"term": term_label(a["kind"], a["value"]), "kind": a["kind"],
                          "salience": a["salience"],
                          "embedding": self.p.embed(term_label(a["kind"], a["value"]),
                                                    dim=self.cfg["embed_dim"])}
                         for a in agg[: self.cfg["max_terms"]]]

        return Profile(structured=structured, text=text, signals=agg,
                       embedding=emb, term_embeddings=term_embs)

    @staticmethod
    def _synthesize_text(s):
        # Mock-Synthese (deterministisch). Im Managed-Betrieb schreibt das LLM diesen Absatz.
        parts = []
        role = s["rolle"] or "researcher"
        if s["forschungsfelder"]:
            parts.append(f"A {role} working on " + ", ".join(s["forschungsfelder"]) + ".")
        if s["methoden"]:
            parts.append("Methods and topics include " + ", ".join(s["methoden"]) + ".")
        if s["sucht"]:
            parts.append("At the conference, looking for " + ", ".join(s["sucht"]) + ".")
        return " ".join(parts)


def term_label(kind, value):
    """Menschlesbares Label eines kanonischen Signals (fuer Anzeige und Term-Embedding)."""
    return {"field": AREA_LABEL, "role": ROLE_LABEL, "seek": SEEK_LABEL}.get(
        kind, {}).get(value, value)


def _struct_to_text(s):
    return " | ".join([
        "fields: " + ", ".join(s["forschungsfelder"]),
        "methods: " + ", ".join(s["methoden"]),
        "role: " + (s["rolle"] or ""),
        "seeking: " + ", ".join(s["sucht"]),
    ])
