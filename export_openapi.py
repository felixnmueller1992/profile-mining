"""Exportiert die OpenAPI-Schnittstellenbeschreibung des Service als YAML.

FastAPI erzeugt die Spezifikation automatisch aus den Routen, Pydantic-Schemas
und Docstrings in service.py; dieses Skript schreibt sie nach openapi.yaml.

Aufruf:  python3 export_openapi.py            (benoetigt: pip install pyyaml)
"""
import os

import yaml

from service import app

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "openapi.yaml")


def main():
    spec = app.openapi()
    with open(OUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    print(f"-> {OUT} ({len(spec['paths'])} Pfade, OpenAPI {spec['openapi']})")


if __name__ == "__main__":
    main()
