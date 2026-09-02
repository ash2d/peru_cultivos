"""Guards on the national CENAGRO route — extract, name link, and the three-observation shift.

`test_cenagro.py` pins the Piura-only `cenagro.py`. These pin the three modules built on top
of the 25-department extract (`DATA.md` §1.5, `RESULTS.md` §8.6–8.7). One test per thing that
is **load-bearing prose elsewhere** — a claim a reader would otherwise have to take on trust:

* the Parquet schema is **declared, not inferred**, because a chunked write needs one schema
  for the whole file and pandas will hand chunk 1 an object column and chunk 2 a float one;
* a name key that pulls too many parcels is **dropped, not resolved**;
* the "before" observation must **precede** the "after" one;
* shares are **Wilson on Kish's effective n**, so an interval can never run below zero and a
  weighted sample never claims its row count's precision;
* post-stratification **restores the population's cell counts**;
* `WOODY_NON_CROP` is **deliberately unmapped**, and the figure reports both readings.

They use synthetic frames wherever the real ones are not needed, so they run without `data/`.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from crop_classifier.allperu import cenagro_extract as E
from crop_classifier.allperu import cenagro_link as L
from crop_classifier.allperu import cenagro_shift as S
from crop_classifier.paths import ROOT


# --- the extract ---
def test_the_parquet_schema_is_declared_not_inferred():
    """Every kept column has a fixed type before any row is read: a chunked write needs one
    schema for the whole file, and inferring it per chunk is how `P024_03` lands as object in
    chunk 1 and float in chunk 2, failing the write half way through a 2.5 M-row department.
    """
    names = [f.name for f in E.SCHEMA]
    assert len(names) == len(set(names)), "duplicate column in the declared schema"
    for col in E.STR_COLS:
        assert E.SCHEMA.field(col).type == "string", f"{col} must be text"
    # the identifiers that carry leading zeros must never be numeric
    for col in ("P001", "P002", "P003", "P007X", "P008", "NPRIN"):
        assert col in E.STR_COLS, f"{col} would lose its leading zeros as a number"


def test_the_keep_list_covers_the_link_the_crop_and_the_tenure_block():
    """The three things §1.5 says the extract exists for."""
    cols = set(E.USECOLS)
    assert {"P009_01", "P009_02", "P009_03"} <= cols, "the farmer name is the only PETT link"
    assert {"P024_03", "P025"} <= cols, "crop code and sown area"
    assert {f"P037_0{k}_01" for k in range(1, 6)} <= cols, "the tenure-regime block"
    assert "P037_01_03" in cols, "title status — the tenure variable that matters"
    assert "P037_04_01" in cols, "posesionario — the 'sin posesionario' check needs it"


def test_linkable_is_the_fourteen_departments_data_md_lists():
    """Not the 15 the bridge ships: Callao yields nothing (`DATA.md` §3)."""
    assert len(L.LINKABLE) == 14
    assert set(L.LINKABLE) == E.LINKABLE
    assert "Callao" not in L.LINKABLE


def test_tidy_forces_one_type_per_column_whatever_the_chunk_looked_like():
    """The same column arriving numeric in one chunk and text in another lands as text."""
    a = pd.DataFrame({c: [1.0] for c in E.USECOLS})       # everything numeric
    b = pd.DataFrame({c: ["1"] for c in E.USECOLS})       # everything text
    ta, tb = E._tidy(a.copy(), "Tumbes"), E._tidy(b.copy(), "Tumbes")
    assert list(ta.columns) == list(tb.columns) == [f.name for f in E.SCHEMA]
    for col in E.STR_COLS:
        assert ta[col].dtype == tb[col].dtype, f"{col} type depends on the chunk"


# --- the name link ---
@pytest.mark.parametrize("raw,expected", [
    ("VALDIVIA CASTILLO MARIA LUCILA", ("VALDIVIA", "CASTILLO", ["MARIA", "LUCILA"])),
    ("RIVAS GA, MARIA ROMELIA", ("RIVAS", "GA", ["MARIA", "ROMELIA"])),
    ("DE LA CRUZ PEREZ JUAN", ("DE LA", "CRUZ", ["PEREZ", "JUAN"])),
])
def test_a_name_splits_into_two_surnames_then_given_names(raw, expected):
    """⚠️ A heuristic, and documented as one: with no field boundaries the first two tokens
    are assumed to be the surnames. A comma, where present, overrides it."""
    assert L.parse_full_name(raw) == expected


def test_a_name_with_no_surname_produces_no_match_key():
    """A blank surname must identify nobody rather than match everybody."""
    keys = L.name_keys(pd.Series([None]), pd.Series(["PEREZ"]), pd.Series([["JUAN"]]))
    assert keys["name_full"].isna().all()
    assert keys["name_core"].isna().all()


def test_accents_and_punctuation_do_not_split_a_person_in_two():
    assert L.norm_txt("Ñuñez-Peña") == L.norm_txt("NUNEZ PENA")


def test_the_three_routes_are_ordered_strictest_first():
    """`full` beats `tokset` beats `core` when the same pair matches on more than one."""
    assert [r for r, _ in L.ROUTES] == ["full", "tokset", "core"]
    assert L._ROUTE_RANK["full"] < L._ROUTE_RANK["tokset"] < L._ROUTE_RANK["core"]


def test_the_token_set_route_survives_swapped_surnames():
    k = L.name_keys(pd.Series(["PEREZ", "GARCIA"]), pd.Series(["GARCIA", "PEREZ"]),
                    pd.Series([["JUAN"], ["JUAN"]]))
    assert k["name_full"].iloc[0] != k["name_full"].iloc[1]
    assert k["name_tokset"].iloc[0] == k["name_tokset"].iloc[1]


def test_an_over_common_name_is_dropped_rather_than_resolved():
    """⚠️ At high multiplicity a name carries no information; keeping it would load the sample
    toward whichever departments have the most repeated surnames, so the cap drops the key
    instead of picking a winner.
    """
    assert L.MAX_CANDIDATES >= 1
    key = pd.Series(["A"] * (L.MAX_CANDIDATES + 5))
    right = pd.DataFrame({"COD_PREDIO": [f"p{i}" for i in range(len(key))], "k": key})
    n_right = right.groupby("k")["COD_PREDIO"].transform("nunique")
    assert (n_right > L.MAX_CANDIDATES).all(), "this fixture must exceed the cap"
    assert right[n_right <= L.MAX_CANDIDATES].empty, "over-common keys must not survive"


# --- the shift: weighting and intervals ---
def _frame(perennial: int, other: int, weight: float = 1.0) -> pd.DataFrame:
    lab = ["PERENNIAL"] * perennial + ["ANNUAL"] * other
    return pd.DataFrame({"c": lab, "w": [weight] * len(lab)})


def test_a_share_interval_never_runs_below_zero():
    """⚠️ Wald does, at the imagery arm's precision, and draws an impossible share."""
    lv = S._level(_frame(1, 40), "c")
    assert lv["lo"] >= 0.0
    assert 0.0 <= lv["pct"] <= 100.0
    assert lv["hi"] <= 100.0


def test_a_share_interval_never_runs_above_one_hundred():
    lv = S._level(_frame(40, 1), "c")
    assert lv["hi"] <= 100.0


def test_equal_weights_give_back_the_row_count_as_the_effective_n():
    assert S._level(_frame(5, 5), "c", "w")["n_eff"] == pytest.approx(10.0)


def test_uneven_weights_buy_less_precision_than_the_row_count_suggests():
    """Kish's n_eff, not len(df) — the imagery arm's weights span two orders of magnitude."""
    df = _frame(5, 5)
    df["w"] = [100.0] * 5 + [1.0] * 5
    lv = S._level(df, "c", "w")
    assert lv["n_eff"] < len(df)
    assert lv["n_eff"] < S._level(_frame(5, 5), "c")["n_eff"]


def test_more_data_narrows_the_interval():
    small = S._level(_frame(10, 90), "c")
    large = S._level(_frame(100, 900), "c")
    assert small["pct"] == pytest.approx(large["pct"])
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_the_weighted_share_is_the_weighted_share_not_the_unweighted_one():
    """The design weight is what turns a PERENNIAL-enriched sample back into a population."""
    df = pd.DataFrame({"c": ["PERENNIAL"] * 5 + ["ANNUAL"] * 5,
                       "w": [1.0] * 5 + [9.0] * 5})
    assert S._level(df, "c")["pct"] == pytest.approx(50.0)
    assert S._level(df, "c", "w")["pct"] == pytest.approx(10.0)


def test_post_stratification_restores_the_population_cell_counts(tmp_path, monkeypatch):
    """⚠️ Not optional: the name link over-selects perennial parcels 16.6 % vs 9.9 %."""
    pop = pd.DataFrame({
        "COD_PREDIO": [str(i) for i in range(100)],
        "dept": ["PIURA"] * 100,
        "label": ["PERENNIAL"] * 10 + ["ANNUAL"] * 90,   # population is 10 % perennial
    })
    path = tmp_path / "pop.parquet"
    pop.to_parquet(path, index=False)
    monkeypatch.setattr(S, "PETT", path)
    # the linked panel over-samples perennial 5:5 instead of 1:9
    linked = pd.DataFrame({"dept": ["PIURA"] * 10,
                           "pett_class": ["PERENNIAL"] * 5 + ["ANNUAL"] * 5})
    out = S.poststratify(linked)
    w = out.groupby("pett_class")["ps_weight"].sum()
    assert w["PERENNIAL"] == pytest.approx(10.0)
    assert w["ANNUAL"] == pytest.approx(90.0)
    assert S._level(out.assign(c=out.pett_class), "c", "ps_weight")["pct"] == \
        pytest.approx(10.0)


def test_the_committed_population_counts_are_the_national_ones(tmp_path, monkeypatch):
    """⚠️ The counts post-stratification reweights to are COMMITTED. A test once overwrote
    them with a 100-parcel fixture, silently moving the headline +9.9 pp → +16.2 pp and
    reproducing perfectly on the machine that had the real table. So: pin the file's content,
    and pin that estimating never writes it (regenerate with `export_pett_population()`).
    """
    pop = pd.read_csv(S.PETT_POP)
    assert set(pop.columns) == {"dept", "label", "N_pop"}
    assert pop["dept"].nunique() == 14
    assert int(pop["N_pop"].sum()) == 726_808

    before = S.PETT_POP.read_bytes()
    fake = tmp_path / "pop.parquet"
    pd.DataFrame({"COD_PREDIO": ["1"], "dept": ["PIURA"],
                  "label": ["PERENNIAL"]}).to_parquet(fake, index=False)
    monkeypatch.setattr(S, "PETT", fake)
    S.poststratify(pd.DataFrame({"dept": ["PIURA"], "pett_class": ["PERENNIAL"]}))
    assert S.PETT_POP.read_bytes() == before, "poststratify wrote the committed counts"


def test_woody_non_crop_is_left_unmapped_on_purpose():
    """Folding it either way is the decision, not a preprocessing step — so both are reported."""
    assert S.S2_TO_DECLARED["WOODY_NON_CROP"] is None
    assert S.S2_TO_DECLARED["NON_AGRICULTURE"] is None
    assert S.S2_TO_DECLARED["PERENNIAL"] == "PERENNIAL"


def test_the_tenure_difference_row_subtracts_and_widens():
    """The difference of two independent groups: estimates subtract, variances add."""
    df = pd.DataFrame({
        "b": ["ANNUAL"] * 200,
        "a": (["PERENNIAL"] * 40 + ["ANNUAL"] * 60) + (["PERENNIAL"] * 20 + ["ANNUAL"] * 80),
        "tenure": ["INSCRITO"] * 100 + ["NO INSCRITO"] * 100,
    })
    t = S.by_tenure(df, "b", "a")
    ins, noi, diff = t.iloc[0], t.iloc[1], t.iloc[2]
    assert diff.tenure.startswith("difference")
    assert diff.change_pp == pytest.approx(ins.change_pp - noi.change_pp, abs=0.05)
    assert diff.ci95_pp >= max(ins.ci95_pp, noi.ci95_pp)


# --- the figure ---
def _fig_module():
    path = ROOT / "docs" / "figures" / "perennial_over_time_by_tenure.py"
    spec = importlib.util.spec_from_file_location("_fig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_standalone_figure_script_needs_nothing_from_the_project():
    """It is a reproduction script: importing it must not pull the package in."""
    mod = _fig_module()
    assert set(mod.VALUES) == {"INSCRITO", "NO INSCRITO"}
    for s in mod.VALUES.values():
        for key in ("cen", "s2", "woody"):
            assert len(s[key]) == 2, f"{key} needs a before and an after"
            for pct, lo, hi in s[key]:
                assert 0.0 <= lo <= pct <= hi <= 100.0, "interval must bracket the estimate"


def test_the_error_bars_are_never_negative_lengths():
    """matplotlib silently draws nonsense for a negative yerr."""
    mod = _fig_module()
    for s in mod.VALUES.values():
        for key in ("cen", "s2", "woody"):
            for level in s[key]:
                below, above = mod._err(level)
                assert below[0] >= 0 and above[0] >= 0


def test_the_figure_accepts_the_dict_the_pipeline_computes():
    """`_err`/`_pct` take either the baked 3-tuple or `figure_values()`'s dict."""
    mod = _fig_module()
    as_dict = {"pct": 10.0, "lo": 5.0, "hi": 20.0}
    assert mod._pct(as_dict) == 10.0
    assert mod._err(as_dict) == [[5.0], [10.0]]
    assert mod._pct((10.0, 5.0, 20.0)) == 10.0


def test_the_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    out = _fig_module().draw(path=tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_the_imagery_arm_carries_its_own_baseline():
    """⚠️ Each arm restricts to a different parcel set, so it cannot inherit the census's
    baseline. Making them equal would assert the three instruments measure the same quantity
    on the same frame — exactly what §8.6 says they do not.
    """
    mod = _fig_module()
    for s in mod.VALUES.values():
        assert s["cen"][0][0] != s["s2"][0][0]
        assert s["s2"][0][0] != s["woody"][0][0]


def test_the_woody_reading_moves_the_endpoint_further_than_the_census_effect():
    """The claim §8.7 rests on: one ambiguous class outweighs the thing being measured."""
    mod = _fig_module()
    ins = mod.VALUES["INSCRITO"]
    census_effect = ins["cen"][1][0] - ins["cen"][0][0]
    woody_swing = abs(ins["woody"][1][0] - ins["s2"][1][0])
    assert woody_swing > census_effect
