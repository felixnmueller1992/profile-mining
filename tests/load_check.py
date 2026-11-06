#!/usr/bin/env python3
"""Lasttest gegen einen echten uvicorn-Prozess (nicht TestClient)."""
import concurrent.futures as cf, os, pathlib, statistics, subprocess, sys, time, urllib.request, json

DIENST = str(pathlib.Path(__file__).resolve().parent.parent)

BASE = "http://127.0.0.1:8099"


def req(method, path, body=None, hdrs=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json", **(hdrs or {})})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def multipart(path, name, content, hdrs=None):
    b = "----X"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: text/plain\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
    r = urllib.request.Request(BASE + path, data=body, method="POST",
                               headers={"Content-Type": f"multipart/form-data; boundary={b}",
                                        **(hdrs or {})})
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "service:app", "--port", "8099",
                        "--host", "127.0.0.1", "--log-level", "error"],
                       cwd=DIENST, env={**os.environ, "PROFILE_MINING_PROVIDER": "mock",
                            # Die globale Jobobergrenze (Voreinstellung 500) ist eine
                            # Schutzgrenze und wird als solche im Unit-Test geprueft
                            # (test_globale_jobobergrenze). Fuer Messung E muss sie
                            # weichen, sonst nimmt der Dienst nur 500 der 5000 Jobs an --
                            # der Test haette dann still etwas anderes gemessen, als
                            # seine Ausgabe behauptet.
                            "MAX_JOBS_TOTAL": "100000"})
for _ in range(60):
    try:
        if req("GET", "/healthz")[0] == 200:
            break
    except Exception:
        time.sleep(0.5)
print("Server laeuft.\n")
DOC = ("Ich bin Doktorandin und arbeite an Deep Learning, Natural Language Processing "
       "und Reinforcement Learning. Methoden: Transformer, Attention, Policy Gradient. "
       "Ich suche Kooperationen und biete Mentoring fuer Nachwuchs. ") * 6

try:
    # --- A: Ist die PATCH-Transition unter echter Parallelitaet atomar? -------
    req("POST", "/profiling", hdrs={"X-User-Id": "race"})
    multipart("/profiling/documents", "a.txt", DOC.encode(), {"X-User-Id": "race"})
    with cf.ThreadPoolExecutor(32) as ex:
        codes = [f.result()[0] for f in [ex.submit(req, "PATCH", "/profiling",
                 {"status": "submitted"}, {"X-User-Id": "race"}) for _ in range(32)]]
    n202 = codes.count(202)
    print(f"A  32 parallele PATCH -> {n202}x 202 Accepted, {codes.count(409)}x 409 Conflict")
    print(f"   {'MEHRFACHSTART MOEGLICH' if n202 > 1 else 'faktisch atomar (GIL); Sperre fehlt dennoch im Code'}\n")

    # --- B: Dokumentgrenze unter echter Parallelitaet -------------------------
    req("POST", "/profiling", hdrs={"X-User-Id": "docs"})
    with cf.ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda i: multipart("/profiling/documents", f"d{i}.txt", b"y" * 500,
                                       {"X-User-Id": "docs"}), range(80)))
    n = len(json.loads(req("GET", "/profiling/documents", hdrs={"X-User-Id": "docs"})[1]))
    print(f"B  80 parallele Uploads bei Grenze 20 -> {n} Dokumente gespeichert")
    print(f"   {'GRENZE UEBERSCHRITTEN' if n > 20 else 'Grenze gehalten'}\n")

    # --- C: Durchsatz kompletter Mining-Laeufe (Mock-Provider, ohne LLM) -----
    def full_run(i):
        u = {"X-User-Id": f"load-{i}"}
        t0 = time.perf_counter()
        req("POST", "/profiling", hdrs=u)
        multipart("/profiling/documents", "cv.txt", DOC.encode(), u)
        req("PATCH", "/profiling", {"status": "submitted"}, u)
        for _ in range(300):
            s = json.loads(req("GET", "/profiling", hdrs=u)[1])["status"]
            if s in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        req("GET", "/profiling/result", hdrs=u)
        return time.perf_counter() - t0, s

    for par in (1, 25, 100):
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(par) as ex:
            res = list(ex.map(full_run, range(par * 4)))
        wall = time.perf_counter() - t0
        lat = sorted(r[0] for r in res)
        ok = sum(1 for r in res if r[1] == "succeeded")
        print(f"C  {par:3d} parallele Nutzer, {len(res)} Laeufe: {wall:5.1f}s Gesamtzeit, "
              f"{len(res)/wall:6.1f} Profile/s, Median {lat[len(lat)//2]*1000:6.0f} ms, "
              f"p95 {lat[int(len(lat)*0.95)]*1000:6.0f} ms, erfolgreich {ok}/{len(res)}")

    # --- D: Threadzahl im Serverprozess nach Last ----------------------------
    with open(f"/proc/{srv.pid}/status") as f:
        thr = [l for l in f if l.startswith("Threads")][0].strip()
    rss = int([l.split()[1] for l in open(f"/proc/{srv.pid}/status") if l.startswith("VmRSS")][0])
    print(f"\nD  Serverprozess nach Last: {thr}, RSS {rss/1024:.0f} MB")

    # --- E: Polling-Kosten bei vielen offenen Jobs ---------------------------
    def mk(i):
        u = {"X-User-Id": f"idle-{i}"}
        return req("POST", "/profiling", hdrs=u)[0]
    with cf.ThreadPoolExecutor(32) as ex:
        codes = list(ex.map(mk, range(5000)))
    angelegt = sum(1 for c in codes if c == 201)
    abgewiesen = sum(1 for c in codes if c == 503)
    lat = []
    for _ in range(40):
        t0 = time.perf_counter(); req("GET", "/profiling", hdrs={"X-User-Id": "idle-0"})
        lat.append(time.perf_counter() - t0)
    print(f"E  GET /profiling bei {angelegt} tatsaechlich offenen Jobs: "
          f"Median {statistics.median(lat)*1000:.2f} ms")
    if abgewiesen:
        print(f"   {abgewiesen} Anlagen mit 503 abgewiesen -- Messung laeuft mit weniger "
              f"Jobs als angenommen, MAX_JOBS_TOTAL pruefen")
finally:
    srv.terminate()
