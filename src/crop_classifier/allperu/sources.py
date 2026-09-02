"""Department -> (SSET crop file, bridge .dta, polygon .shp) registry for all of Peru.

The 2026-08-07 drop added the rest of Peru alongside the Piura files, in three new folders —
``data/raw/BD_SSET/`` (8 multi-department xlsx), ``data/raw/Grafica_Tabular/`` (one .dta per
department), ``data/raw/QGIS/<DEPT>/`` (one shapefile per department). The Piura members were
verified **byte-identical** (sha256) to the ones the Piura pipeline used, so this module
supersedes ``build_training_data``'s hard-coded paths without changing any Piura result.

Two things about the new folders that this module exists to paper over:

1. **Most department shapefiles ship their attribute table under the wrong basename** —
   ``QGIS/ANCASH/ANCASH.dbf`` beside ``CATASTRO_..._ANCASH_..._FINAL.shp``. GDAL opens the
   ``.shp`` without it and returns *zero* columns, so the join key silently vanishes.
   :func:`shapefile_view` symlinks consistent basenames; record counts checked to match per
   department.
2. **Only 15 of 23 departments have a bridge**, and it is mandatory: BD SSET has no
   ``COD_PREDIO``, the shapefiles no ``CodigoSSET``. (Callao then links nothing, so 14 carry
   the data.) Departments without a bridge (Amazonas, Apurimac, Cusco, Huanuco, Junin, Madre
   de Dios, Puno, Ucayali — plus Loreto and San Martin, no polygons either) **cannot** be
   linked and are excluded.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pyogrio

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
D_SSET = RAW / "BD_SSET"
D_BRIDGE = RAW / "Grafica_Tabular"
D_QGIS = RAW / "QGIS"

# Sidecars to link alongside the .shp. .dbf is handled separately (see the docstring).
_SIDECARS = (".shp", ".shx", ".prj", ".sbn", ".sbx", ".qix", ".cpg")

# Arequipa ships the same 135,780 parcels twice, UTM 18S and 19S. Take 18S: it carries its
# own correctly-named .dbf, and everything is reprojected to 4326 downstream anyway.
_PREFER_ZONE = {"AREQUIPA": "Z18S"}


# BD SSET's `DEPARTAMENTO` doesn't always hold the department name — Callao is under its
# constitutional-province title, so a plain match finds zero Callao rows and it drops silently.
_DEPT_ALIASES = {
    "PROV.CONST.DEL CALLAO": "CALLAO",
    "PROV CONST DEL CALLAO": "CALLAO",
    "LIMA METROPOLITANA": "LIMA",
}


def _norm(s: str) -> str:
    """Uppercase, unaccented, underscores->spaces, aliases resolved — the three sources
    disagree on all of those, so every department name goes through here before matching.
    """
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = " ".join(s.upper().replace("_", " ").split())
    return _DEPT_ALIASES.get(s, s)


@dataclass(frozen=True)
class Dept:
    """One department's three linked sources."""

    name: str            # canonical QGIS folder name, e.g. "LA_LIBERTAD"
    sset: Path           # multi-department BD SSET workbook containing it
    bridge: Path         # Grafica_Tabular/<Dept>.dta
    shp: Path            # QGIS/<DEPT>/CATASTRO_..._FINAL.shp (original location)

    @property
    def sset_key(self) -> str:
        """Value to match against the workbook's ``DEPARTAMENTO`` column."""
        return _norm(self.name)


def _sset_index() -> dict[str, Path]:
    """``normalised department -> workbook``, parsed from the ``BD SSET(A-B-C).xlsx`` names."""
    out: dict[str, Path] = {}
    for f in sorted(D_SSET.glob("BD SSET(*.xlsx")):
        inner = f.stem[f.stem.index("(") + 1: f.stem.rindex(")")]
        for part in inner.split("-"):
            out[_norm(part)] = f
    return out


def _bridge_index() -> dict[str, Path]:
    return {_norm(p.stem): p for p in sorted(D_BRIDGE.glob("*.dta"))}


def _shp_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in sorted(D_QGIS.iterdir()):
        if not d.is_dir():
            continue
        cands = sorted(d.glob("CATASTRO_*_FINAL.shp"))
        if not cands:
            continue
        if len(cands) > 1:
            zone = _PREFER_ZONE.get(d.name)
            cands = [c for c in cands if zone and zone in c.stem] or cands[:1]
        out[_norm(d.name)] = cands[0]
    return out


def departments() -> list[Dept]:
    """Every department linkable end to end, i.e. present in **all three** sources.

    Excludes the eight polygon-only departments and the two (Loreto, San Martin) that have
    SSET crops but neither a bridge nor polygons — see the module docstring.
    """
    ssets, bridges, shps = _sset_index(), _bridge_index(), _shp_index()
    out = []
    for key in sorted(set(bridges) & set(shps) & set(ssets)):
        folder = next(d.name for d in D_QGIS.iterdir()
                      if d.is_dir() and _norm(d.name) == key)
        out.append(Dept(name=folder, sset=ssets[key], bridge=bridges[key],
                        shp=shps[key]))
    return out


def unlinkable() -> dict[str, list[str]]:
    """What each source has that the other two do not — the exclusion audit trail."""
    ssets, bridges, shps = _sset_index(), _bridge_index(), _shp_index()
    return {"polygons_but_no_bridge": sorted(set(shps) - set(bridges)),
            "sset_but_no_polygons": sorted(set(ssets) - set(shps)),
            "bridge_but_no_polygons": sorted(set(bridges) - set(shps))}


def shapefile_view(dept: Dept, view_dir: Path) -> Path:
    """Symlink ``dept``'s shapefile into ``view_dir`` with a consistent basename + a .dbf.

    Returns the path to link at. Raw data is never modified. Raises if no attribute table
    can be found at all, because the failure mode otherwise is a silent zero-column read.
    """
    view_dir.mkdir(parents=True, exist_ok=True)
    stem, base = dept.shp.stem, dept.shp.with_suffix("")
    for ext in _SIDECARS:
        src = base.with_suffix(ext)
        if src.exists():
            _link(src, view_dir / f"{stem}{ext}")

    dbf = base.with_suffix(".dbf")
    if not dbf.exists():
        cands = [p for p in dept.shp.parent.glob("*.dbf")
                 if not p.name.startswith("CATASTRO")]
        if not cands:
            raise FileNotFoundError(
                f"{dept.name}: no .dbf for {dept.shp.name} — the attribute table (and so "
                f"COD_PREDIO) is missing; GDAL would read this as zero columns")
        dbf = cands[0]
    _link(dbf, view_dir / f"{stem}.dbf")
    return view_dir / f"{stem}.shp"


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def shp_info(path: Path) -> dict:
    """``{n_features, crs, fields}`` without reading geometry."""
    info = pyogrio.read_info(str(path))
    return {"n_features": int(info["features"]), "crs": str(info["crs"]),
            "fields": list(info["fields"])}
