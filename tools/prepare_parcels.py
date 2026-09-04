"""Thin shim: the module moved into the package so `cc` can reuse it.

    uv run python tools/prepare_parcels.py my_farms.shp --year 2020 --out data/mine
"""
from crop_classifier.prepare_parcels import main

if __name__ == "__main__":
    main()
