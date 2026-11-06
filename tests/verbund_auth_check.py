"""Prueft die Token-Pruefung vor dem Dienst im Verbund (Traefik + Keycloak + Backend).

Der Dienst liest aus dem Bearer-Token nur den sub-Claim und prueft die Signatur
nicht. Das darf er nur, weil die Traefik-Route jede Anfrage vorher an GET /auth/me
des Backends schickt (ForwardAuth, siehe traefik_profilemining.yaml). Dieses Skript
prueft von aussen, ob das im laufenden Verbund tatsaechlich so ist:

  A  ohne Token                                   -> 401
  B  Token ohne Signatur (alg=none), fremder sub  -> 401
  C  echtes Token, sub umgeschrieben              -> 401  (Signatur passt nicht mehr)
  D  wie C, an /embeddings                        -> 401  (die Route deckt beide Pfade)
  E  echtes Token                                 -> Job anlegen und lesen (201, 200)
  F  echtes Token + fremde X-User-Id              -> derselbe Job wie in E
  G  nur X-User-Id, kein Token                    -> 401
  H  CORS-Vorabanfrage des Browsers (OPTIONS)     -> 2xx mit Allow-Origin
  I  abgelaufenes Token (optional, --abgelaufen)  -> 401
  J  Dienst am Proxy vorbei (optional, --direkt)  -> nicht erreichbar

Die Faelle B und C sind die eigentliche Probe. Ohne die Pruefung vor dem Dienst
nimmt der Dienst beide an, denn er liest den sub-Claim, ohne die Signatur
anzusehen. A und G allein belegen nichts: Ohne Kennung antwortet auch der Dienst
selbst mit 401 (STRICT_AUTH=1). Die Spalte "Quelle" zeigt deshalb, wer
geantwortet hat. Der Dienst antwortet als application/problem+json, das Backend
nicht.

Das Skript verbraucht kein Kontingent der Grenze je Identitaet, weil es keinen Job
einreicht. Den in E angelegten Entwurf loescht es am Ende wieder.

Aufruf (Token frisch holen, es lebt fuenf Minuten; siehe Verbundtest_Checkliste.md):

    python tests/verbund_auth_check.py --base https://docker.localhost/api --token "eyJ..."
        [--direkt http://localhost:8080] [--abgelaufen "eyJ..."] [--protokoll datei.md]

Rueckgabewert 0, wenn alle ausgefuehrten Faelle bestehen, sonst 1.
Braucht nur httpx (steht in requirements.txt).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from urllib.parse import urlsplit

import httpx

FREMD = "fremde-kennung-verbund-auth-check"


# ------------------------------------------------------------- Token-Basteleien ---

def _b64(daten: bytes) -> str:
    return base64.urlsafe_b64encode(daten).rstrip(b"=").decode()


def _unb64(teil: str) -> bytes:
    return base64.urlsafe_b64decode(teil + "=" * (-len(teil) % 4))


def teile(token: str) -> tuple[dict, dict, str]:
    kopf, nutzlast, signatur = token.split(".")
    return json.loads(_unb64(kopf)), json.loads(_unb64(nutzlast)), signatur


def ohne_signatur(token: str, sub: str) -> str:
    """Gleiche Claims, anderer sub, alg=none und leere Signatur."""
    _, nutzlast, _ = teile(token)
    nutzlast["sub"] = sub
    kopf = {"alg": "none", "typ": "JWT"}
    return f"{_b64(json.dumps(kopf).encode())}.{_b64(json.dumps(nutzlast).encode())}."


def sub_umgeschrieben(token: str, sub: str) -> str:
    """Echter Kopf und echte Signatur, aber eine veraenderte Nutzlast."""
    kopf, nutzlast, signatur = token.split(".")
    claims = json.loads(_unb64(nutzlast))
    claims["sub"] = sub
    return f"{kopf}.{_b64(json.dumps(claims).encode())}.{signatur}"


# ----------------------------------------------------------------- Auswertung ---

class Fall:
    def __init__(self, kurz: str, titel: str, erwartet: str):
        self.kurz, self.titel, self.erwartet = kurz, titel, erwartet
        self.ok: bool | None = None
        self.ist = ""
        self.quelle = ""

    def setze(self, ok: bool, ist: str, quelle: str = "") -> None:
        self.ok, self.ist, self.quelle = ok, ist, quelle


def quelle(antwort: httpx.Response) -> str:
    art = antwort.headers.get("content-type", "")
    if "problem+json" in art:
        return "Dienst"
    if antwort.status_code in (401, 403):
        return "Proxy/Backend"
    return "-"


def kurz(antwort: httpx.Response) -> str:
    return f"{antwort.status_code}"


def pruefen(a: argparse.Namespace) -> list[Fall]:
    basis = a.base.rstrip("/")
    origin = a.origin or "{0.scheme}://{0.netloc}".format(urlsplit(basis))
    c = httpx.Client(base_url=basis, timeout=20.0, verify=False)
    echt = {"Authorization": f"Bearer {a.token}"}
    claims = teile(a.token)[1]
    eigener_sub = claims.get("sub", "?")
    faelle: list[Fall] = []

    def fall(kurzname, titel, erwartet) -> Fall:
        f = Fall(kurzname, titel, erwartet)
        faelle.append(f)
        return f

    # A -- ohne Token
    f = fall("A", "ohne Token", "401")
    r = c.get("/profiling")
    f.setze(r.status_code == 401, kurz(r), quelle(r))

    # B -- alg=none
    f = fall("B", "Token ohne Signatur (alg=none), fremder sub", "401 vom Proxy")
    r = c.get("/profiling", headers={"Authorization": f"Bearer {ohne_signatur(a.token, FREMD)}"})
    f.setze(r.status_code == 401 and quelle(r) != "Dienst", kurz(r), quelle(r))

    # C -- sub umgeschrieben
    gefaelscht = sub_umgeschrieben(a.token, FREMD)
    f = fall("C", "echtes Token, sub umgeschrieben", "401 vom Proxy")
    r = c.get("/profiling", headers={"Authorization": f"Bearer {gefaelscht}"})
    f.setze(r.status_code == 401 and quelle(r) != "Dienst", kurz(r), quelle(r))

    # D -- dasselbe an /embeddings
    f = fall("D", "wie C, an /embeddings", "401 vom Proxy")
    r = c.post("/embeddings", json={"description": "dense retrieval"},
               headers={"Authorization": f"Bearer {gefaelscht}"})
    f.setze(r.status_code == 401 and quelle(r) != "Dienst", kurz(r), quelle(r))

    # E -- echtes Token: Entwurf anlegen und lesen
    f = fall("E", "echtes Token: Job anlegen und lesen", "201, dann 200")
    c.delete("/profiling", headers=echt)            # Rest eines frueheren Laufs
    r1 = c.post("/profiling", headers=echt)
    r2 = c.get("/profiling", headers=echt)
    job_id = r2.json().get("id") if r2.status_code == 200 else None
    f.setze(r1.status_code == 201 and r2.status_code == 200,
            f"{r1.status_code}, {r2.status_code}", quelle(r2))
    if not f.ok and r1.status_code == 403:
        f.ist += " (Token ohne Rolle 'user', etwa ein Admin-Konto?)"

    # F -- echtes Token, dazu fremde X-User-Id
    f = fall("F", "echtes Token + fremde X-User-Id", "200, derselbe Job wie E")
    r = c.get("/profiling", headers={**echt, "X-User-Id": FREMD})
    gleich = r.status_code == 200 and job_id is not None and r.json().get("id") == job_id
    ist = f"{r.status_code}"
    if r.status_code == 404:
        ist += " (Header wurde als Identitaet genommen)"
    f.setze(gleich, ist, quelle(r))

    # G -- nur X-User-Id
    f = fall("G", "nur X-User-Id, kein Token", "401")
    r = c.get("/profiling", headers={"X-User-Id": FREMD})
    f.setze(r.status_code == 401, kurz(r), quelle(r))

    # H -- CORS-Vorabanfrage
    f = fall("H", "CORS-Vorabanfrage (OPTIONS)", "2xx, Allow-Origin gesetzt")
    r = c.request("OPTIONS", "/profiling", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type"})
    erlaubt = r.headers.get("access-control-allow-origin", "")
    f.setze(200 <= r.status_code < 300 and erlaubt == origin,
            f"{r.status_code}, Allow-Origin={erlaubt or '-'}", quelle(r))

    # I -- abgelaufenes Token (optional)
    if a.abgelaufen:
        f = fall("I", "abgelaufenes Token", "401 vom Proxy")
        r = c.get("/profiling", headers={"Authorization": f"Bearer {a.abgelaufen}"})
        f.setze(r.status_code == 401 and quelle(r) != "Dienst", kurz(r), quelle(r))

    # J -- am Proxy vorbei (optional)
    if a.direkt:
        f = fall("J", f"Dienst direkt ({a.direkt})", "nicht erreichbar")
        try:
            r = httpx.get(a.direkt.rstrip("/") + "/healthz", timeout=5.0, verify=False)
            f.setze(False, f"{r.status_code} (Dienst antwortet am Proxy vorbei)", "Dienst")
        except httpx.HTTPError as e:
            f.setze(True, type(e).__name__)

    # Aufraeumen: den Entwurf aus E wieder loeschen
    c.delete("/profiling", headers=echt)
    c.close()
    a._sub = eigener_sub
    return faelle


def ausgeben(faelle: list[Fall], a: argparse.Namespace) -> str:
    zeilen = [
        f"# Token-Pruefung im Verbund, {datetime.now():%d.%m.%Y %H:%M}",
        "",
        f"Basis: `{a.base}` · sub des echten Tokens: `{getattr(a, '_sub', '?')}`",
        "",
        "| Fall | Probe | Erwartet | Ist | Quelle | |",
        "|---|---|---|---|---|---|",
    ]
    for f in faelle:
        marke = "bestanden" if f.ok else "**FEHLER**"
        zeilen.append(f"| {f.kurz} | {f.titel} | {f.erwartet} | {f.ist} | {f.quelle or '-'} | {marke} |")
    n_ok = sum(1 for f in faelle if f.ok)
    zeilen += ["", f"{n_ok} von {len(faelle)} Faellen bestanden."]
    if not all(f.ok for f in faelle):
        zeilen += ["", "Hinweise: B/C/D vom Dienst beantwortet heisst, die Route traegt die "
                       "ForwardAuth-Middleware nicht (Proxy neu starten, unter Windows "
                       "bemerkt der Watcher Dateiaenderungen nicht). F mit 404 heisst, der "
                       "Dienst laeuft mit ALLOW_DEV_IDENTITY=1 und der Proxy entfernt "
                       "X-User-Id nicht."]
    return "\n".join(zeilen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="z. B. https://docker.localhost/api")
    ap.add_argument("--token", required=True, help="frisches Keycloak-Token (Rolle user)")
    ap.add_argument("--origin", help="Origin fuer die CORS-Probe; Vorgabe: Schema und Host von --base")
    ap.add_argument("--abgelaufen", help="ein abgelaufenes Token derselben Instanz (optional)")
    ap.add_argument("--direkt", help="Host-Adresse des Dienstes am Proxy vorbei (optional)")
    ap.add_argument("--protokoll", help="Ergebnis zusaetzlich als Markdown-Datei ablegen")
    a = ap.parse_args()
    try:
        teile(a.token)
    except Exception:
        print("--token ist kein JWT (drei Base64-Teile, durch Punkte getrennt).")
        return 2
    import warnings
    warnings.filterwarnings("ignore")             # mkcert-Zertifikat, verify=False
    try:
        faelle = pruefen(a)
    except httpx.HTTPError as e:
        print(f"Verbund nicht erreichbar unter {a.base}: {e}")
        return 2
    text = ausgeben(faelle, a)
    print(text)
    if a.protokoll:
        with open(a.protokoll, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if all(f.ok for f in faelle) else 1


if __name__ == "__main__":
    sys.exit(main())
