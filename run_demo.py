#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo: mined ein Profil aus Beispiel-Quellen (ohne API, MockProvider) und zeigt
strukturierte + Freitext-Repraesentation, belegte Top-Signale und Kompressions-Effekt."""
import json, glob, os
from provider import MockProvider
from pipeline import ProfileMiner, Source
from embedding_compression import matryoshka_truncate, quantize, cosine, hamming_sim

def load_sources():
    srcs = []
    for path in sorted(glob.glob(os.path.join("sample_docs", "*.txt"))):
        name = os.path.basename(path)
        is_own = "read_note" not in name           # gelesene Fremdarbeit = niedrige Autoritaet
        srcs.append(Source(id=name, text=open(path, encoding="utf-8").read(), is_own=is_own))
    return srcs

def main():
    miner = ProfileMiner(MockProvider(), cfg={"representation": "text"})
    prof = miner.mine(load_sources())

    print("=== STRUKTURIERTES PROFIL (Studie 3.1, Variante A) ===")
    print(json.dumps(prof.structured, ensure_ascii=False, indent=2))
    print("\n=== FREITEXT-PROFIL (Studie 3.1, Variante B) ===")
    print(prof.text)

    print("\n=== TOP-SIGNALE MIT BELEG (Anti-Halluzination) ===")
    for a in prof.signals[:6]:
        print(f"  [{a['kind']:6} {a['value']:>14}] salience={a['salience']:<5} "
              f"quellen={a['n_sources']}  beleg: \"{a['evidence'][:70]}...\"")

    print("\n=== EMBEDDING + KOMPRESSION (Studie 3.2) ===")
    full = prof.embedding
    t256 = full
    t128 = matryoshka_truncate(full, 128)
    _, b_full = quantize(full, "float32")
    bits, b_bin = quantize(full, "binary")
    print(f"  Dimension voll={len(full)} (float32, {b_full} B)  ->  "
          f"trunc128 ({len(t128)} dim)  ->  binaer ({b_bin} B)")
    print(f"  Cosine(voll, trunc128) = {cosine(full[:128], t128):.3f}  "
          f"(Qualitaetsabfall durch Kompression -> in Studie 3.2 systematisch gemessen)")

if __name__ == "__main__":
    main()
