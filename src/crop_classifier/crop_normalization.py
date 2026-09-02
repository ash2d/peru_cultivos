"""Normalise the free-text ``CULTIVO`` crop labels from the BD SSET registry.

~9,400 distinct dirty labels in Piura — typos, accents, plurals, regional synonyms,
land-prep/fallow states, and multiple crops packed into one cell with a dozen separators.
Turns one raw label into a list of ``(crop, category)`` pairs, nothing silently dropped.

Conservative by design: split on every seen separator (``, / + & - Y CON`` + percentage
weightings, parentheticals stripped); normalise spelling not identity (``CANON`` fixes
typos/accents/plurals/synonyms but never merges distinct crops — the bean varieties stay
separate); nothing removed (fallow / land_prep / pasture / unspecified survive, flagged by
``category``); unknown tokens pass through cleaned, defaulting to ``crop`` unless a keyword
marks them otherwise.
"""

from __future__ import annotations

import re

# --- 1 · Low-level text cleaning ---

# Strip vowel accents but KEEP "Ñ" (meaningful: CAÑA = sugarcane).
_ACCENTS = str.maketrans("ÁÀÄÂÉÈËÊÍÌÏÎÓÒÖÔÚÙÜÛ", "AAAAEEEEIIIIOOOOUUUU")

# Crop separators. "Y"/"CON" only as whole words (so GUAYAQUIL, SOYA, YUCA are not split).
# "E" is deliberately not a separator — too risky ("EN DESCANSO", "DE") for the 3 rows it helps.
_SPLIT = re.compile(r"[,/+&]|-+|\bY\b|\bCON\b")

# Descriptive prefixes wrapping a crop noun, stripped repeatedly:
# "EXISTEN RASTROJOS DE MAIZ" -> "MAIZ".
_PREFIX = re.compile(
    r"^(CULTIVO DE|CULTIVO|SEMBRIO DE|SEMBRADO DE|SEMBRADO CON|SEMBRADO|SEMBRIO|"
    r"RASTROJOS DE|RASTROJO DE|RASTROJOS|RASTROJO|RESTOS DE|"
    r"EXISTEN|EXISTE|HAY|SOLO|SOLAMENTE|ACTUALMENTE|"
    r"PARA SIEMBRA DE|PARA SIEMBRA|SIEMBRA DE|EN LIMPIEZA PARA SIEMBRA DE|"
    r"EN|CON|DE)\s+"
)


def clean_text(s: str) -> str:
    """Uppercase, de-accent, collapse whitespace, and turn percentage weightings /
    parentheticals into comma separators — crops here are often delimited only by their
    share (``PLATANO (30%) CAFE (30%)``, ``CAFE 50%-PLATANO 50%``), so ``(...)`` groups and
    digit-runs become commas for the splitter.
    """
    s = str(s).upper().translate(_ACCENTS)
    s = re.sub(r"\(.*?\)", ",", s)      # "(50%)" -> separator
    s = re.sub(r"\d+\s*%?", ",", s)     # "50%", "50", bare number -> separator
    s = s.replace("%", ",")
    s = re.sub(r"[.;:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- 2 · Canonical synonym / typo map (cleaned variant -> canonical) ---
# Only spelling variants, accents, plurals and documented regional synonyms — never two
# distinct crops.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    # --- staple annual crops ---
    "ARROZ": ("AROZ", "ARROS", "ARROZAL", "ARRZ"),
    "MAIZ": ("MAIZ AMARILLO", "MAICES", "MAIZAL", "MAIZ CHALA", "MASISO"),
    "ALGODON": ("ALGODONERO", "ALGODONAL", "ALGODONERA", "ALODON"),
    "TRIGO": ("TRIGAL",),
    "CEBADA": (),
    "FRIJOL": ("FREJOL", "FRIXOL", "FRIJOLES", "FREJOLES", "FRJOL"),
    "FRIJOL DE PALO": ("FREJOL DE PALO", "FRIJOL PALO", "FREJOL PALO"),
    "FRIJOL CASTILLA": ("FREJOL CASTILLA",),
    "FRIJOL CAUPI": ("FREJOL CAUPI", "FRIJOL CHILENO", "FREJOL CHILENO"),
    "ZARANDAJA": ("SARANDAJA", "SARANDAJO", "ZARANDAJO", "CHILENA SARANDAJA"),
    "ARVEJA": ("ALVERJA", "ALBERJA", "ARVERJA", "ARBEJA", "ALBERJAS", "ARVEJAS"),
    "HABA": ("HABAS",),
    "LENTEJA": ("LENTEJAS",),
    "SOYA": ("SOJA",),
    "MANI": ("MANIS",),
    "PALLAR": ("PALLARES",),
    # --- roots / tubers / vegetables ---
    "YUCA": ("YUCAL",),
    "PAPA": ("PAPAS",),
    "CAMOTE": ("CAMOTES",),
    "OLLUCO": ("OLLUCOS", "ULLUCO"),
    "OCA": (),
    "ZANAHORIA": ("ZANAHORIAS",),
    "CEBOLLA": ("CEBOLLAS",),
    "TOMATE": ("TOMATES",),
    "ZAPALLO": ("ZAPALLOS",),
    "REPOLLO": ("REPOLLOS", "COL"),
    "AJO": ("AJOS",),
    # --- industrial / other annuals ---
    "CAÑA DE AZUCAR": ("CAÑA", "CANA DE AZUCAR", "CANA", "CAÑAVERAL", "CAÑA AZUCAR"),
    "GIRASOL": ("GIRASOLES",),
    "SORGO": (),
    "MARIGOLD": ("MARIGOL", "MARYGOL", "MARIGOLT"),
    "TABACO": (),
    # --- fruit trees / perennials ---
    "MANGO": ("MANGOS", "MANGAL", "MANGO CIRUELO"),
    "LIMON": ("LIMONERO", "LIMON SUTIL", "LIMONEROS", "LIMONERA", "LIMONES"),
    "NARANJA": ("NARANJO", "NARANJOS", "NARANJAL", "NARANJILLA"),
    "PLATANO": ("PLATANOS", "PLATANO SEDA", "PLATANO ISLA", "PLATANAL"),
    "BANANO": ("BANANOS", "BANANO ORGANICO", "GUINEO", "GUINEO CHOLO"),
    "CAFE": ("CAFETO", "CAFETAL", "CAFETOS", "CAFEE"),
    "CACAO": ("CACAOTAL", "CACAOTERO"),
    "PALTA": ("PALTO", "PALTOS", "PALTAS"),
    "PAPAYA": ("PAPAYO", "PAPAYOS"),
    "COCO": ("COCOTERO", "COCOTEROS", "COCOS"),
    "CIRUELA": ("CIRUELO", "CIRUELOS", "CIRUELAS"),
    "GUANABANA": ("GUANABANO",),
    "GRANADILLA": ("GRANADILLAS",),
    "MARACUYA": ("MARACUYAS",),
    "TAMARINDO": ("TAMARINDOS",),
    "ZAPOTE": ("ZAPOTES",),
    "GUABO": ("HUABO", "GUABA", "HUABA", "PACAE", "PACAY", "GUAMO"),
    "HIGUERON": ("HIGUERA", "HIGO", "HIGOS"),
    "UVA": ("UVAS", "VID", "PARRA"),
    "TUNA": ("TUNAS",),
    "MAMEY": ("MAMEYES",),
    "CHIRIMOYA": ("CHIRIMOYO", "CHIRIMOYOS"),
    # --- forestry / agroforestry ---
    "EUCALIPTO": ("EUCALIPTOS", "EUCALITO"),
    "PINO": ("PINOS",),
    "ALGARROBO": ("ALGARROBOS", "ALGARROVO"),
    "FAIQUE": ("FAIQUES", "FAIKE"),
    "GUAYAQUIL": ("GUAYAQUILL", "CAÑA GUAYAQUIL", "BAMBU"),
    "ROBLE": ("ROBLES",),
    "CIPRES": ("CIPRESES",),
    "NOGAL": ("NOGALES",),
    "ALISO": ("ALISOS",),
    "ALCANFOR": ("ALCANFORES",),
    # --- pasture / fodder grasses (category = pasture) ---
    "PASTO": ("PASTOS", "PASTO CULTIVADO", "PASTO SEMBRADO"),
    "PASTO NATURAL": ("PASTOS NATURALES", "NATURALES", "PASTO NATURALES"),
    "PASTO ELEFANTE": ("PASTO DE ELEFANTE", "ELEFANTE", "PASTO ELFANTE", "PASTO ELEFANTES"),
    "MELQUERON": ("MERKERON", "MERQUERON", "MELKERON", "PASTO MELQUERON",
                  "PASTO MELKERON", "PASTO MERKERON", "MELQUERON PASTO"),
    "GRAMA": ("GRAMAS",),
    "GRAMALOTE": ("GRAMALOTES",),
    "GRAMA CRIOLLA": ("GRAMA CRIOLLO",),
    "REYGRAS": ("RYE GRASS", "RAY GRASS", "REY GRASS", "RYEGRASS", "RYGRASS", "REYGRASS"),
    "INVERNA": ("PASTO INVERNA", "PAJA DE INVERNA", "IMBERNA"),
    "NUDILLO": ("GRAMA NUDILLO", "NUDILLOS"),
    "YARAGUA": ("YARAGUAL", "PASTO YARAGUA"),
    "GRAMINEAS": ("GRAMINEA",),
    "ALFALFA": ("ALFALFAL", "ALFA ALFA"),
    "ALFALFILLA": (),
    # --- generic / unspecified vegetation groups (kept, category = crop) ---
    "FRUTALES": ("FRUTAL", "ARBOLES FRUTALES", "FRUTA"),
    "HORTALIZAS": ("HORTALIZA",),
    "MENESTRAS": ("MENESTRA", "LEGUMINOSAS", "LEGUMINOSA"),
    "FLORES": ("FLOR",),
    "PAN LLEVAR": ("PANLLEVAR", "PAN DE LLEVAR", "PAN LLEBAR"),
    "COBERTURA ARBOREA": ("COBERTURA ARBOLES", "COBERTURA BOSCOSA", "BOSQUE"),
}

# Fallow / land-preparation states written in the crop field. Kept, categorised so a
# classifier can exclude them.
_FALLOW: dict[str, tuple[str, ...]] = {
    "DESCANSO": ("EN DESCANSO", "EN DESCANZO", "DESCANZO", "TERRENO EN DESCANSO",
                 "TERRENO EN DESCANZO", "PREDIO EN DESCANSO", "PREDIO EN DESCANZO",
                 "TERRENO DESCANSO", "EN DECANSO", "DESCANSO ", "EN DESCANSO EXISTEN "),
    "BARBECHO": ("TERRENO EN BARBECHO", "EN BARBECHO", "BARVECHO"),
    "ERIAZO": ("TERRENO ERIAZO", "ERIAZOS", "ERIZO"),
    "PURMA": ("PURMAS", "MONTE", "MONTES", "MONTE PURMA"),
    "ABANDONO": ("EN ABANDONO", "ABANDONADO", "TERRENO ABANDONADO"),
    "TERRENO SALINO": ("SALINO", "TERRENO SALITROSO", "SALITROSO"),
}

_LAND_PREP: dict[str, tuple[str, ...]] = {
    "MECANIZADO": ("MECANIZADA", "MECANISADO", "MECANIZACION", "NO MECANIZADO", "MECANIZ"),
    "GRADEO": ("GRADEADO", "TERRENO GRADEADO", "GRADEA", "EN GRADEO"),
    "ARADO": ("ARADURA", "ARADO DE TERRENO", "TERRENO ARADO"),
    "NIVELACION": ("NIVELADO", "NIVELACION Y GRADEO", "NIVELANDO"),
    "LIMPIEZA": ("EN LIMPIEZA", "LIMPIEZA DE TERRENO", "LIMPIA"),
    "PREPARACION": ("PREPARACION DEL TERRENO", "EN PREPARACION", "PREPARANDO",
                    "PREPARACION DE TERRENO", "PREPARADO"),
    "DESHIERBO": ("DESHIERBO QUEMA", "DESHIERBE", "DESYERBO"),
    "QUEMA": ("QUEMADO", "ROZO Y QUEMA"),
    "ROZO": ("ROSO", "EN ROZO"),
    "REFORESTACION": ("REFORESTADO", "EN REFORESTACION"),
}

_UNSPECIFIED: dict[str, tuple[str, ...]] = {
    "SIN ESPECIFICAR": ("NO ESPECIFICA", "NO ESPECIFICADO", "SIN ESPECIFICACION",
                        "NO INDICA", "NO PRECISA", "CULTIVO", "SIN DATO", "S/C",
                        "SIN CULTIVO", "SIN USO", "NINGUNO", "OTROS", "VARIOS"),
}

# Categories, in priority order (first match wins when building the reverse index).
_CATEGORY_SOURCES: list[tuple[str, dict[str, tuple[str, ...]]]] = [
    ("fallow", _FALLOW),
    ("land_prep", _LAND_PREP),
    ("unspecified", _UNSPECIFIED),
    ("crop", _SYNONYMS),  # pasture tokens are re-tagged below
]

# Pasture canonical tokens get their own category — a valid land-cover class, distinct from
# row crops.
_PASTURE_CANON = {
    "PASTO", "PASTO NATURAL", "PASTO ELEFANTE", "MELQUERON", "GRAMA", "GRAMALOTE",
    "GRAMA CRIOLLA", "REYGRAS", "INVERNA", "NUDILLO", "YARAGUA", "GRAMINEAS",
    "ALFALFA", "ALFALFILLA",
}

# Keyword fallbacks for tokens not in any map above (long tail). Checked as substrings.
_KW_FALLOW = ("DESCAN", "DESCANZ", "BARBECHO", "BARVECHO", "ERIAZO", "ERIZO",
              "ABANDON", "PURMA", "SALIN", "SALITR", "SIN CULTIVO", "SIN USO")
_KW_LAND_PREP = ("MECANIZ", "GRADE", "ARAD", "NIVELAC", "LIMPIEZA", "PREPARAC",
                 "DESHIERB", "DESYERB", "QUEMA", "ROZO", "REFOREST", "SEMBRIO",
                 "PARA SIEMBRA", "GRADEO")
_KW_PASTURE = ("PASTO", "GRAMA", "PAJA", "REYGRAS", "RYE", "GRAMINEA", "INVERNA",
               "MELQUER", "MERKER", "NUDILLO", "YARAGUA", "ELEFANTE", "ZACATE")
_KW_UNSPEC = ("NO ESPECIF", "NO INDICA", "NO PRECISA", "SIN DATO", "DESCONOCIDO")


def _build_reverse_index() -> tuple[dict[str, str], dict[str, str]]:
    """Return (variant -> canonical, canonical -> category)."""
    to_canon: dict[str, str] = {}
    to_category: dict[str, str] = {}
    for category, source in _CATEGORY_SOURCES:
        for canon, variants in source.items():
            cat = "pasture" if (category == "crop" and canon in _PASTURE_CANON) else category
            to_canon[canon] = canon
            to_category[canon] = cat
            for v in variants:
                to_canon[clean_text(v)] = canon
    return to_canon, to_category


_TO_CANON, _TO_CATEGORY = _build_reverse_index()


def _keyword_category(token: str) -> str:
    """Category for a token that is not in the curated maps (long-tail fallback)."""
    for kw in _KW_FALLOW:
        if kw in token:
            return "fallow"
    for kw in _KW_LAND_PREP:
        if kw in token:
            return "land_prep"
    for kw in _KW_UNSPEC:
        if kw in token:
            return "unspecified"
    for kw in _KW_PASTURE:
        if kw in token:
            return "pasture"
    return "crop"


def _lookup(token: str) -> tuple[str, str] | None:
    """Exact map lookup for a single already-cleaned token (with a plural-``S`` retry).
    Returns ``(canonical, category)`` if the token is *known*, else ``None``."""
    if token in _TO_CANON:
        canon = _TO_CANON[token]
        return canon, _TO_CATEGORY[canon]
    if token.endswith("S") and token[:-1] in _TO_CANON:
        canon = _TO_CANON[token[:-1]]
        return canon, _TO_CATEGORY[canon]
    return None


def canonicalize_token(token: str) -> list[tuple[str, str]]:
    """Map one cleaned sub-string to a list of ``(canonical_name, category)`` pairs.

    Usually one; a space-separated run of *known* crops (``MANGO LIMON``) expands to
    several. ``[]`` for empty / noise input.
    """
    prev = None
    while token != prev:                        # strip nested prefixes
        prev = token
        token = _PREFIX.sub("", token, count=1).strip()
    token = re.sub(r"\s+", " ", token)
    if not token or len(token) < 2:
        return []

    hit = _lookup(token)                         # whole token known (multi-word crops win)
    if hit is not None:
        return [hit]

    # unknown multi-word: split on spaces only if every piece is a known crop
    # ("MANGO LIMON" -> split; "MAIZ AMARILLO DURO" stays whole)
    pieces = token.split(" ")
    if len(pieces) > 1:
        resolved = [_lookup(p) for p in pieces]
        if all(r is not None for r in resolved):
            return [r for r in resolved if r is not None]

    # long-tail passthrough: keep the cleaned token, category from keywords
    return [(token, _keyword_category(token))]


def normalize_label(raw: str) -> list[tuple[str, str]]:
    """Turn one raw ``CULTIVO`` cell into a list of ``(crop, category)`` pairs.

    Order-preserving, de-duplicated within the label. Returns ``[]`` for empty/NaN input.
    """
    if raw is None or isinstance(raw, float):
        return []
    cleaned = clean_text(raw)
    if not cleaned:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for piece in _SPLIT.split(cleaned):
        for canon, category in canonicalize_token(piece):
            if canon not in seen:
                seen.add(canon)
                out.append((canon, category))
    return out
