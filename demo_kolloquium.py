#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vorfuehrskript fuer das Kolloquium.

Spielt den vollstaendigen Vertrag des Profilmining-Dienstes gegen einen laufenden
Dienst durch und protokolliert jeden Schritt gross und lesbar. Gedacht als zweite
Stufe hinter der Swagger-Oberflaeche: wenn Klicken in der Bildschirmfreigabe hakt,
laeuft hier derselbe Ablauf in einem Rutsch.

    python demo_kolloquium.py                     # gegen http://localhost:8090
    python demo_kolloquium.py --base https://docker.localhost/api --token "eyJ..."
    python demo_kolloquium.py --nur vertrag       # ohne den Sicherheitsteil
    python demo_kolloquium.py --pause             # nach jedem Abschnitt auf Enter warten

Der Sicherheitsteil verwendet bewusst eigene Kennungen, damit die Ratengrenze der
Vorfuehrungskennung unangetastet bleibt. Im Verbund (mit --token) gibt es keine
eigenen Kennungen mehr: Der Proxy laesst nur gepruefte Tokens durch. Dort zeigt
Abschnitt 10 deshalb, dass ein gefaelschtes Token vor dem Dienst abgewiesen wird.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import httpx

# ---------------------------------------------------------------- Darstellung

BREIT = 78
FARBE = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _f(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if FARBE else text


def abschnitt(nr: str, titel: str) -> None:
    print()
    print(_f("36;1", "=" * BREIT))
    print(_f("36;1", f" {nr}  {titel}"))
    print(_f("36;1", "=" * BREIT))


def erwartung(text: str) -> None:
    print(_f("90", f"   erwartet: {text}"))


def zeige(antwort: httpx.Response, zeilen: int = 6, feld: str | None = None) -> None:
    """Eine Antwort knapp und lesbar ausgeben."""
    ok = antwort.status_code < 400
    marke = _f("32;1", f"{antwort.status_code} {antwort.reason_phrase}") if ok \
        else _f("33;1", f"{antwort.status_code} {antwort.reason_phrase}")
    print(f"   {_f('1', antwort.request.method):<6} {antwort.request.url.path:<28} -> {marke}")
    for kopf in ("retry-after", "content-type"):
        if kopf in antwort.headers and antwort.status_code >= 400:
            print(_f("90", f"          {kopf}: {antwort.headers[kopf]}"))
    if not antwort.content:
        return
    try:
        daten = antwort.json()
    except ValueError:
        print(_f("90", "          " + antwort.text[:200]))
        return
    if feld and isinstance(daten, dict) and feld in daten:
        daten = {feld: daten[feld]}
    text = json.dumps(daten, ensure_ascii=False, indent=2)
    for i, zeile in enumerate(text.splitlines()):
        if i >= zeilen:
            print(_f("90", f"          ... ({len(text.splitlines()) - zeilen} weitere Zeilen)"))
            break
        print(_f("90", "          " + zeile[:120]))


def kuerzen(vektor: list, n: int = 4) -> str:
    kopf = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in vektor[:n])
    return f"[{kopf}, ... ] ({len(vektor)} Werte)"


# ---------------------------------------------------------------- Ablauf

class Demo:
    def __init__(self, basis: str, kennung: str, token: str | None, pause: bool):
        self.basis = basis.rstrip("/")
        self.pause = pause
        self.verbund = bool(token)     # im Verbund liegt nur ein Teil der Pfade auf der Route
        self.token = token
        kopf = {"X-User-Id": kennung}
        if token:
            kopf = {"Authorization": f"Bearer {token}"}
        # verify=False: der Verbund benutzt lokal ausgestellte mkcert-Zertifikate
        # (docker.localhost). Das ist eine Vorfuehrhilfe, kein Produktionsmuster.
        self.c = httpx.Client(base_url=self.basis, headers=kopf, timeout=30.0, verify=False)

    def halt(self) -> None:
        if self.pause:
            input(_f("90", "\n   [Enter]"))

    # -- Teil 1: der Vertrag ------------------------------------------------

    def vertrag(self, docs: list[str]) -> None:
        abschnitt("1", "Der Dienst lebt")
        erwartung("200 mit Provider und Praefix")
        a = self.c.get("/healthz")
        if a.status_code == 200 and "provider" in a.text:
            zeige(a)
        else:
            # Im Verbund routet Traefik nur /profiling und /embeddings hierher;
            # /healthz landet beim Backend nebenan. Das ist kein Fehler.
            print(_f("90", f"   GET    /healthz                     -> "
                           f"{a.status_code} (nicht auf der Traefik-Route)"))
            print(_f("90", "          Im Verbund liegen nur /profiling und /embeddings auf der"))
            print(_f("90", "          Route des Dienstes; er meldet sich mit dem ersten Job."))
        self.halt()

        abschnitt("2", "Job anlegen - und der Job ist ein Singleton")
        erwartung("erst 201 draft, dann 409 - je Identitaet genau ein Job")
        zeige(self.c.post("/profiling"), feld="status")
        zeige(self.c.post("/profiling"), zeilen=4)
        self.halt()

        abschnitt("3", "Quellen anhaengen")
        erwartung("je 201; eigene Arbeiten und Gelesenes werden unterschieden")
        for pfad in docs:
            # Gelesene Fremdarbeit (read_note) als solche kennzeichnen, sonst
            # gilt sie per Voreinstellung als eigene Arbeit (is_own=True).
            name = os.path.basename(pfad)
            meta = json.dumps({"is_own": "read_note" not in name})
            with open(pfad, "rb") as f:
                zeige(self.c.post("/profiling/documents",
                                  files={"file": (name, f, "text/plain")},
                                  data={"metadata": meta}),
                      zeilen=4)
        self.halt()

        abschnitt("4", "Einreichen - der Dienst antwortet sofort")
        erwartung("202 Accepted; das Mining laeuft asynchron weiter")
        zeige(self.c.patch("/profiling", json={"status": "submitted"}), zeilen=5)
        self.halt()

        abschnitt("5", "Fortschritt abfragen")
        erwartung("Polling bis succeeded; Retry-After steuert den Takt")
        for versuch in range(60):
            a = self.c.get("/profiling")
            d = a.json()
            print(f"   Abruf {versuch + 1:>2}: status={_f('1', str(d.get('status')))} "
                  f"progress={d.get('progress')} stage={d.get('stage')}")
            if d.get("status") in ("succeeded", "failed"):
                break
            time.sleep(1.0)
        self.halt()

        abschnitt("6", "Ergebnis abholen - genau einmal")
        erwartung("erst 200 mit beiden Repraesentationen, dann 410 Gone")
        a = self.c.get("/profiling/result")
        if a.status_code == 200:
            d = a.json()
            profil = d.get("profile", d)
            print(f"   {_f('1', 'GET')}    /profiling/result            -> "
                  f"{_f('32;1', '200 OK')}")
            beschr = profil.get("description") or ""
            print()
            print(_f("1", "   Freitext-Repraesentation"))
            for zeile in _umbruch(beschr, BREIT - 6):
                print("      " + zeile)
            struktur = profil.get("structured") or {}
            if struktur:
                print()
                print(_f("1", "   Strukturierte Repraesentation (Auszug)"))
                for schl in list(struktur)[:5]:
                    wert = struktur[schl]
                    if isinstance(wert, list):
                        wert = ", ".join(str(x) for x in wert[:5])
                    print(f"      {schl:<22} {str(wert)[:90]}")
            signale = profil.get("signals") or d.get("signals") or []
            if signale:
                print()
                print(_f("1", "   Belegte Signale - jedes mit woertlichem Zitat"))
                for s in signale[:4]:
                    beleg = str(s.get("evidence", ""))[:64]
                    print(f"      {str(s.get('kind','')):<8} {str(s.get('value',''))[:24]:<26} "
                          f"sal={s.get('salience')} q={s.get('n_sources')}")
                    print(_f("90", f"         „{beleg}...“"))
            vektor = profil.get("embedding") or d.get("embedding")
            if isinstance(vektor, list):
                print()
                print(_f("1", "   Embedding fuers geraeteseitige Matching"))
                print("      " + kuerzen(vektor))
        else:
            zeige(a)
        print()
        erwartung("zweiter Abruf: 410 - das Ergebnis ist serverseitig weg")
        zeige(self.c.get("/profiling/result"), zeilen=4)
        self.halt()

        abschnitt("7", "Aufraeumen und der zustandslose Endpunkt")
        erwartung("204 beim Loeschen; /embeddings speichert ohnehin nichts")
        zeige(self.c.delete("/profiling"))
        a = self.c.post("/embeddings",
                        json={"description": "I work on federated learning and "
                                             "privacy-preserving machine learning "
                                             "for mobile devices."})
        if a.status_code == 200:
            d = a.json()
            print(f"   {_f('1', 'POST')}   /embeddings                  -> {_f('32;1', '200 OK')}")
            print(f"      areas_of_interest: {', '.join(d.get('areas_of_interest', []))}")
            print(f"      embedding:         {kuerzen(d.get('embedding', []))}")
        else:
            zeige(a)
        self.halt()

    # -- Teil 2: die Grenzen ------------------------------------------------

    def sicherheit(self, kennung: str) -> None:
        abschnitt("8", "Ohne Kennung geht nichts (sichere Voreinstellung)")
        if self.verbund:
            erwartung("401 - im Verbund weist schon der Proxy ab, der Dienst sieht nichts")
        else:
            erwartung("401 als application/problem+json - STRICT_AUTH steht auf 1")
        with httpx.Client(base_url=self.basis, timeout=15.0, verify=False) as anonym:
            zeige(anonym.get("/profiling"), zeilen=5)
        self.halt()

        abschnitt("9", "Grenze am Embedding-Endpunkt")
        erwartung("413 - der Endpunkt kostet Modellaufrufe, also hat er eine Grenze")
        zeige(self.c.post("/embeddings", json={"description": "x" * 25_000}), zeilen=5)
        self.halt()

        if self.verbund:
            self.gefaelschtes_token()
            return

        abschnitt("10", "Grenze je Identitaet")
        erwartung("die ersten Einreichungen 202, dann 429 mit Retry-After")
        with httpx.Client(base_url=self.basis, headers={"X-User-Id": kennung},
                          timeout=30.0, verify=False) as vielnutzer:
            for i in range(1, 8):
                vielnutzer.delete("/profiling")   # Rest aus einer Probe stoert sonst
                vielnutzer.post("/profiling")
                vielnutzer.post("/profiling/documents",   # ohne Quelle kein Einreichen
                                files={"file": ("last.txt", b"I work on dense retrieval.",
                                                "text/plain")})
                a = vielnutzer.patch("/profiling", json={"status": "submitted"})
                marke = _f("32;1", str(a.status_code)) if a.status_code < 400 \
                    else _f("33;1", str(a.status_code))
                zusatz = ""
                if a.status_code == 429:
                    zusatz = _f("90", f"   Retry-After: {a.headers.get('retry-after')} s")
                print(f"   Einreichung {i}: {marke}{zusatz}")
                if a.status_code == 429:
                    print()
                    print(_f("90", "   " + json.dumps(a.json(), ensure_ascii=False)[:160]))
                    break
                vielnutzer.delete("/profiling")
        self.halt()


    def gefaelschtes_token(self) -> None:
        """Nur im Verbund: dasselbe Token mit umgeschriebenem sub-Claim.

        Der Dienst selbst liest den sub-Claim, ohne die Signatur zu pruefen. Dass er
        das darf, liegt an der Pruefung davor (ForwardAuth auf /auth/me). Die Grenze
        je Identitaet laesst sich hier nicht mehr mit einer Dev-Kennung zeigen, und
        mit dem echten Token waere das Kontingent fuer eine Stunde verbraucht.
        """
        abschnitt("10", "Gefaelschtes Token: sub-Claim umgeschrieben")
        erwartung("401 vom Proxy - die Signatur passt nicht mehr, der Dienst sieht nichts")
        kopf, nutzlast, signatur = self.token.split(".")
        claims = json.loads(base64.urlsafe_b64decode(nutzlast + "=" * (-len(nutzlast) % 4)))
        claims["sub"] = "jemand-anderes"
        neu = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
        with httpx.Client(base_url=self.basis, timeout=15.0, verify=False,
                          headers={"Authorization": f"Bearer {kopf}.{neu}.{signatur}"}) as f:
            zeige(f.get("/profiling"), zeilen=5)
        self.halt()


def _umbruch(text: str, breite: int) -> list[str]:
    worte, zeilen, aktuell = text.split(), [], ""
    for w in worte:
        if len(aktuell) + len(w) + 1 > breite:
            zeilen.append(aktuell)
            aktuell = w
        else:
            aktuell = f"{aktuell} {w}".strip()
    if aktuell:
        zeilen.append(aktuell)
    return zeilen


def main() -> int:
    ap = argparse.ArgumentParser(description="Vorfuehrung des Profilmining-Dienstes")
    ap.add_argument("--base", default=os.environ.get("DEMO_BASE", "http://localhost:8090"),
                    help="Basis-URL, im Verbund z. B. https://docker.localhost/api")
    ap.add_argument("--token", default=os.environ.get("DEMO_TOKEN"),
                    help="Keycloak-Bearer-Token; ohne das wird X-User-Id gesetzt")
    # Wichtig: die Kennung bekommt von sich aus einen Zeitstempel. Der Dienst laesst
    # je Identitaet fuenf Einreichungen pro Stunde zu -- mit einer festen Kennung
    # waere nach der fuenften Probe die Vorfuehrung selbst mit 429 abgewiesen worden.
    ap.add_argument("--user", default=f"kolloquium-{time.strftime('%H%M%S')}",
                    help="Kennung fuer die Vorfuehrung (Vorgabe: mit Zeitstempel, "
                         "damit Proben die Ratengrenze der Vorfuehrung nicht aufbrauchen)")
    ap.add_argument("--nur", choices=["vertrag", "sicherheit"], help="nur einen Teil zeigen")
    ap.add_argument("--pause", action="store_true", help="nach jedem Abschnitt auf Enter warten")
    ap.add_argument("--docs", nargs="*", default=None, help="Quelldateien (Vorgabe: sample_docs)")
    a = ap.parse_args()

    docs = a.docs
    if docs is None:
        ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
        docs = sorted(os.path.join(ordner, n) for n in os.listdir(ordner) if n.endswith(".txt"))

    print(_f("1", "\n  Profilmining-Dienst - Vorfuehrung"))
    print(_f("90", f"  Basis: {a.base}   Kennung: "
                   f"{'Keycloak-Token' if a.token else a.user}   Quellen: {len(docs)}"))

    d = Demo(a.base, a.user, a.token, a.pause)
    try:
        d.c.get("/healthz")
    except httpx.HTTPError as e:
        print(_f("31;1", f"\n  Dienst nicht erreichbar unter {a.base}: {e}"))
        print(_f("90", "  Starten mit:  python -m uvicorn service:app --port 8090"))
        return 2

    # Vorfuehrungskennung sauber halten: ein Rest-Job aus der Probe stoert das Bild.
    d.c.delete("/profiling")

    if a.nur != "sicherheit":
        d.vertrag(docs)
    if a.nur != "vertrag":
        d.sicherheit(a.user + "-last")

    print()
    print(_f("36;1", "=" * BREIT))
    print(_f("1", "  Fertig. Vertrag vollstaendig, Grenzen gezeigt."))
    print(_f("90", "  96 Tests sichern denselben Ablauf ab; die Schnittstellen-"
                   "beschreibung unter /docs\n  wird aus demselben Code erzeugt."))
    print(_f("36;1", "=" * BREIT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
