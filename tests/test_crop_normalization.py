"""Pin the crop-label normalisation decisions (see crop_normalization.py docstring).

These cases are the tricky ones the cleaner is designed to get right; they double as
executable documentation of what the pipeline does to the raw ``CULTIVO`` field.
"""

import pytest

from crop_classifier.crop_normalization import normalize_label


def crops(label: str) -> list[str]:
    return [c for c, _ in normalize_label(label)]


def cats(label: str) -> list[str]:
    return [cat for _, cat in normalize_label(label)]


@pytest.mark.parametrize(
    "label, expected",
    [
        # single crops, typos and accents fixed
        ("ARROZ", ["ARROZ"]),
        ("AROZ", ["ARROZ"]),
        ("CAFÉ", ["CAFE"]),
        ("ALGODONERO", ["ALGODON"]),          # -ero variant -> base crop
        ("LIMON SUTIL", ["LIMON"]),           # default lime variety collapsed
        # intercrops: every separator style must split into a list
        ("CAFE Y PLATANO", ["CAFE", "PLATANO"]),
        ("MANGO, LIMON", ["MANGO", "LIMON"]),
        ("MANGO LIMON", ["MANGO", "LIMON"]),          # bare space, both known crops
        ("CAFE 50%-PLATANO 50%", ["CAFE", "PLATANO"]),
        ("CAFE 50, PLATANO 50", ["CAFE", "PLATANO"]),  # numbers-as-percent, no % sign
        ("CAFE(50%), PLATANO(50%)", ["CAFE", "PLATANO"]),
        (
            "PLATANO (30%) CAFE (30%) NARANJA (20%) GUAYAQUIL (10%)",
            ["PLATANO", "CAFE", "NARANJA", "GUAYAQUIL"],
        ),
        # descriptive prefixes stripped (nested too)
        ("CULTIVO DE ALGODONERO", ["ALGODON"]),
        ("EN DESCANSO, EXISTEN RASTROJOS DE ARROZ", ["DESCANSO", "ARROZ"]),
    ],
)
def test_crop_lists(label, expected):
    assert crops(label) == expected


def test_multiword_crops_are_not_over_split():
    # A multi-word crop must survive as one token, not be split on its space.
    assert crops("CAÑA DE AZUCAR") == ["CAÑA DE AZUCAR"]
    assert crops("FRIJOL DE PALO") == ["FRIJOL DE PALO"]
    assert crops("MAIZ AMARILLO DURO") == ["MAIZ AMARILLO DURO"]
    assert crops("GRAMA CRIOLLA") == ["GRAMA CRIOLLA"]


def test_categories_flag_non_crops_without_dropping():
    # Fallow / land-prep / pasture are kept but tagged so they can be filtered downstream.
    assert cats("TERRENO EN DESCANZO") == ["fallow"]
    assert cats("DESHIERBO QUEMA") == ["land_prep"]
    assert cats("PASTO ELEFANTE") == ["pasture"]
    assert cats("ARROZ") == ["crop"]


def test_empty_and_noise():
    assert normalize_label(None) == []
    assert normalize_label(float("nan")) == []
    assert normalize_label("   ") == []


def test_dedup_within_label():
    # A crop repeated inside one cell appears once.
    assert crops("MAIZ, MAIZ") == ["MAIZ"]
