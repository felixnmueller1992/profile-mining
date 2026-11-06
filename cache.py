#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Result-Cache mit Single-Read-Semantik und TTL (Referenzimplementierung, In-Memory).

Bildet den kurzlebigen serverseitigen Zustand des Dienstes ohne dauerhafte
Profilspeicherung ab (Abschnitt 5.4/6.5):
Ein Profiling-Ergebnis ist genau EINMAL abrufbar (Read-and-Evict) und verfaellt
nach einer kurzen Frist. In einer mehrinstanzigen Bereitstellung tritt ein externer
Cache (Redis mit GETDEL + TTL) an die Stelle dieser Klasse.
"""
import time, secrets


class SingleReadExpired(Exception):
    pass


class ResultCache:
    def __init__(self, ttl_seconds=3600, _clock=time.time):
        self.ttl = ttl_seconds
        self._clock = _clock
        self._store = {}  # token -> (value, expires_at)
        # Fruehester Zeitpunkt, zu dem ein Eintrag verfallen sein kann. Davor ist
        # ein Sweep beweisbar ergebnislos und wird uebersprungen -- dieselbe exakte
        # Schranke wie beim Job-Verfall (service.sweep_jobs). Sie darf nur zu frueh
        # liegen, nie zu spaet: Entfernte Eintraege machen sie hoechstens veraltet,
        # dann laeuft ein Scan zu viel.
        self._next_expiry = float("inf")

    def put(self, value) -> str:
        token = "res-" + secrets.token_hex(8)
        expires_at = self._clock() + self.ttl
        self._store[token] = (value, expires_at)
        self._next_expiry = min(self._next_expiry, expires_at)
        return token

    def read_once(self, token):
        """Atomar lesen UND loeschen. Zweiter Aufruf -> SingleReadExpired."""
        self._sweep()
        if token not in self._store:
            raise SingleReadExpired(token)
        value, _ = self._store.pop(token)   # evict on read
        return value

    def exists(self, token) -> bool:
        self._sweep()
        return token in self._store

    def discard(self, token) -> None:
        """Entfernt einen Eintrag, ohne ihn zu lesen (der zugehoerige Job ist weg).

        Ohne diesen Weg blieb ein nicht abgeholtes Ergebnis nach DELETE /profiling
        bis zu seiner Frist im Speicher -- nicht mehr abrufbar, aber vorhanden.
        """
        if token:
            self._store.pop(token, None)

    def sweep(self) -> int:
        """Raeumt abgelaufene Eintraege ab; Rueckgabe: Anzahl der entfernten.

        Oeffentlich, damit der Dienst auch ohne Lesezugriff aufraeumt. Nur beim
        Lesen zu raeumen tut im Lastfall "viele warten, niemand holt ab" nichts.
        """
        return self._sweep()

    def _sweep(self) -> int:
        now = self._clock()
        if now < self._next_expiry:
            return 0
        stale = [t for t, (_, exp) in self._store.items() if exp <= now]
        for t in stale:
            del self._store[t]
        self._next_expiry = min((exp for _, exp in self._store.values()),
                                default=float("inf"))
        return len(stale)
