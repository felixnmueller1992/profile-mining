#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding-Kompression fuer den Bluetooth-Mesh-Austausch (FF3).

Hinweis: Frueher hiess dieses Modul compression.py; es wurde umbenannt, weil
Python 3.14 ein gleichnamiges Standardbibliotheks-Paket eingefuehrt hat, das
sonst beim Import von torch/gzip ueberdeckt wuerde.
"""
import math


def l2norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def matryoshka_truncate(vec, dim):
    """Erste dim Komponenten + Re-Normalisierung (Matryoshka-Annahme).

    dim <= 0 bedeutet: nicht kuerzen, die native Dimension des Modells behalten.
    Damit laesst sich "voller Vektor" ausdruecken, ohne die Modellgroesse zu
    kennen -- noetig, weil der API-Vertrag ausdruecklich festhaelt, dass die
    Dimension modellabhaengig ist und nicht fest verdrahtet werden darf.
    """
    if dim <= 0 or dim >= len(vec):
        return list(vec)
    return l2norm(vec[:dim])


def quantize(vec, mode="float32"):
    """mode: float32 | int8 | binary. Gibt (werte, bytes_pro_vektor) zurueck."""
    d = len(vec)
    if mode == "float32":
        return list(vec), d * 4
    if mode == "int8":
        q = [max(-127, min(127, round(x * 127))) for x in vec]
        return q, d * 1
    if mode == "binary":
        bits = [1 if x >= 0 else 0 for x in vec]
        return bits, math.ceil(d / 8)
    raise ValueError(mode)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # erwartet L2-normalisierte Vektoren


def hamming_sim(a_bits, b_bits):
    same = sum(1 for x, y in zip(a_bits, b_bits) if x == y)
    return same / len(a_bits) if a_bits else 0.0
