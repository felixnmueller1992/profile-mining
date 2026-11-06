#!/usr/bin/env python3
"""Prueft den PRODUKTIVEN Pfad (ManagedLLMProvider): Aufrufzahl je Profil,
Wirkung und Korrektheit der prozessweiten Drosselung, Prompt-Injektion."""
import os, sys, threading, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import provider as P
from pipeline import ProfileMiner, Source

DOC = ("Ich bin Doktorandin und arbeite an Deep Learning, Natural Language Processing "
       "und Reinforcement Learning. Methoden: Transformer, Attention, Policy Gradient. "
       "Ich suche Kooperationen und biete Mentoring fuer den Nachwuchs. ")

# --- 1: Wie viele LLM-Aufrufe kostet EIN Profil? ------------------------------
class Counting(P.ManagedLLMProvider):
    def __init__(self, **kw):
        os.environ["LLM_BASE_URL"] = "http://x"; os.environ["LLM_MODEL"] = "m"
        super().__init__(**kw); self.n = 0
    def _chat(self, prompt):
        self.n += 1
        return '{"signals":[{"kind":"field","value":"dl","evidence":"Deep Learning","confidence":0.9}]}'
    def embed(self, text, dim=0, quant="float32"):
        return [0.1] * 384

print("1  LLM-Aufrufe je Profil (Chunkgroesse 1200 Zeichen):")
for kb, label in ((2, "kurzes Abstract"), (10, "Lebenslauf"), (40, "Paper (PDF)"),
                  (2000, "20 Dokumente am Limit (40 MB)")):
    p = Counting()
    src = [Source(id="d", text=DOC * (kb * 1024 // len(DOC) + 1))]
    ProfileMiner(p, cfg={"term_embeddings": False}).mine(src)
    print(f"     {kb:5d} KB Quelltext ({label:32s}) -> {p.n:5d} Aufrufe")

# --- 2: Greift die Drosselung unter Parallelitaet? ----------------------------
print("\n2  Prozessweite Drosselung (_min_interval als Klassenattribut):")
P.ManagedLLMProvider._min_interval = 1.0
P.ManagedLLMProvider._last_request = 0.0
zeiten = []
lock = threading.Lock()
class Timed(Counting):
    def _post(self, url, headers, payload):
        self._throttle()
        with lock:
            zeiten.append(time.time())
        type(self)._last_request = time.time()
        return {"choices": [{"message": {"content": "{}"}}]}
    def _chat(self, prompt):
        self._post("", {}, {})
        return '{"signals":[]}'

ts = [threading.Thread(target=lambda: Timed()._chat("x")) for _ in range(10)]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
zeiten.sort()
abstaende = [round(zeiten[i + 1] - zeiten[i], 3) for i in range(len(zeiten) - 1)]
verletzt = sum(1 for a in abstaende if a < 0.9)
print(f"     10 parallele Aufrufe bei Mindestabstand 1.0 s")
print(f"     Gesamtdauer {time.time()-t0:.2f} s (bei korrekter Serialisierung ~9 s)")
print(f"     Abstaende: {abstaende}")
print(f"     -> {verletzt} von {len(abstaende)} Abstaenden unter dem Soll: "
      f"{'Drosselung greift bei Parallelitaet NICHT (Wettlauf auf _last_request)' if verletzt else 'ok'}")

print("\n3  Serielle Wirkung derselben Drosselung (1 Thread, 5 Aufrufe):")
P.ManagedLLMProvider._last_request = 0.0
p = Timed(); t0 = time.time()
for _ in range(5):
    p._chat("x")
print(f"     {time.time()-t0:.1f} s fuer 5 Aufrufe -> Obergrenze "
      f"{1/P.ManagedLLMProvider._min_interval:.1f} Aufrufe/s FUER DEN GANZEN PROZESS, "
      f"unabhaengig von der Nutzerzahl")

# --- 4: Prompt-Injektion in der Voreinstellung -------------------------------
print("\n4  Prompt-Injektion aus einem hochgeladenen Dokument:")
class Injected(Counting):
    """Simuliert ein Modell, das der Anweisung im Dokument folgt und Werte samt
    Beleg frei erfindet (der Beleg steht NICHT im Quelltext)."""
    def _chat(self, prompt):
        self.n += 1
        return ('{"signals":[{"kind":"role","value":"prof","evidence":"Ich bin Professor",'
                '"confidence":1.0},{"kind":"seek","value":"hiring","evidence":"we are hiring",'
                '"confidence":1.0}]}')

harmlos = "Ich bin Masterstudent und interessiere mich fuer Datenbanken."
for verify in (False, True):
    p = Injected(verify_evidence=verify)
    sig = p.extract_signals(harmlos + " " * 40, "de")
    art = "Voreinstellung (verify_evidence=False)" if not verify else "verify_evidence=True"
    print(f"     {art:40s} -> {len(sig)} erfundene Signale uebernommen: "
          f"{[(s.kind, s.value) for s in sig]}")
print("     -> In der Voreinstellung wird nur geprueft, DASS ein Belegfeld gefuellt ist,")
print("        nicht, dass der Beleg im Quelltext steht.")
