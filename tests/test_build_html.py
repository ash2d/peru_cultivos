"""Unit tests for the labelling HTML (docs/s2_labelling/plan.md).

**The blindness assertion is the important one and it asserts on the raw string.** It is
not enough that the declared class is un-rendered: if it is anywhere in the embedded JSON
a curious labeller can read it, and a labeller anchored on the 1998 declaration
manufactures agreement between declaration and endpoint — the human form of the
``centroid_lat`` failure this project has already measured three times.
"""

from __future__ import annotations

import base64
import io
import json

import pandas as pd
import pytest
from PIL import Image

from crop_classifier.labelling.build_html import (
    CODEBOOK_HTML,
    FORBIDDEN_FIELDS,
    ITEM_FIELDS,
    ITEMS_CLOSE,
    ITEMS_OPEN,
    LABEL_DISPLAY,
    LABELS,
    LANGS,
    UI,
    build,
    build_items,
    render_html,
    shard,
    trace_svg,
)


def _jpeg(path, size=(320, 320)):
    Image.new("RGB", size, (40, 90, 40)).save(path, "JPEG", quality=72)


@pytest.fixture
def campaign(tmp_path):
    n = 6
    chips = tmp_path / "chips"
    chips.mkdir()
    sample = pd.DataFrame({
        "item_id": [f"M{i:05d}" for i in range(n)],
        "COD_PREDIO": [f"P{i}" for i in range(n)],
        "dept": ["LA_LIBERTAD"] * n,
        "declared_class": ["PERENNIAL"] * 3 + ["ANNUAL"] * 3,
        "crop_set": ["MANGO"] * n,
        "stratum": ["LA_LIBERTAD|PERENNIAL"] * n,
        "region_id": [f"r{i}" for i in range(n)],
        "split": ["test"] * 2 + ["trainval"] * 4,
        "fold": [-1, -1, 0, 1, 2, 3],
        "weight": [123.45] * n,
        "batch": ["main"] * n,
        "overlap": [False] * n,
        "area_ha": [1.5] * n,
        "imagery_date": ["2022-06-15"] * n,
        "imagery_res": ["1.2m"] * n,
    })
    for i in range(n):
        _jpeg(chips / f"M{i:05d}_context.jpg")
        _jpeg(chips / f"M{i:05d}_zoom.jpg")
    px = pd.DataFrame([
        {"COD_PREDIO": f"P{i}", "date": d, "n_px": 40.0, "eroded": True,
         "B": 900.0, "G": 1200.0, "R": 1100.0, "NIR": 3200.0,
         "SWIR1": 2400.0, "SWIR2": 1800.0,
         "NDVI_px_p25": 0.42, "NDVI_px_p50": 0.49, "NDVI_px_p75": 0.57}
        for i in range(n)
        for d in pd.date_range("2021-06-15", "2023-06-15", freq="30D").strftime("%Y-%m-%d")
    ])
    return sample, chips, px, tmp_path / "html"


class TestTraceSvg:
    def test_is_valid_inline_svg_with_a_point_per_observation(self):
        dates = list(pd.date_range("2021-06-15", "2023-06-15", freq="60D")
                     .strftime("%Y-%m-%d"))
        svg = trace_svg(dates, [0.4] * len(dates), "2022-06-15")
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert svg.count("<circle") == len(dates)

    def test_y_axis_is_fixed_so_parcels_are_comparable(self):
        """A flat parcel and a peaking one must not both fill the panel."""
        flat = trace_svg(["2022-01-01", "2022-06-01"], [0.2, 0.2], "2022-06-15")
        peak = trace_svg(["2022-01-01", "2022-06-01"], [0.1, 0.9], "2022-06-15")
        y_flat = {c.split('cy="')[1].split('"')[0]
                  for c in flat.split("<circle")[1:]}
        y_peak = {c.split('cy="')[1].split('"')[0]
                  for c in peak.split("<circle")[1:]}
        assert len(y_flat) == 1 and len(y_peak) == 2

    def test_handles_a_parcel_with_no_observations(self):
        svg = trace_svg([], [], "2022-06-15")
        assert svg.startswith("<svg") and "<circle" not in svg


class TestTraceRibbon:
    """The p25-p75 band: the spread of NDVI across the parcel's own pixels on each date."""

    def test_a_ribbon_is_drawn_when_quantiles_are_supplied(self):
        dates = list(pd.date_range("2021-06-15", "2023-06-15", freq="60D")
                     .strftime("%Y-%m-%d"))
        n = len(dates)
        svg = trace_svg(dates, [0.5] * n, "2022-06-15",
                        lo=[0.4] * n, hi=[0.6] * n)
        assert "<path" in svg and "fill-opacity" in svg

    def test_no_quantiles_still_renders__the_page_degrades_not_fails(self):
        """A store extracted before the quartiles existed must still build a shard."""
        dates = ["2022-01-01", "2022-06-01"]
        svg = trace_svg(dates, [0.2, 0.6], "2022-06-15")
        assert svg.startswith("<svg") and "<path" not in svg
        assert svg.count("<circle") == 2

    def test_the_ribbon_is_behind_the_median_line_and_its_points(self):
        """A band painted over the line would hide the series it qualifies."""
        dates = ["2022-01-01", "2022-06-01"]
        svg = trace_svg(dates, [0.5, 0.5], "2022-06-15", lo=[0.4, 0.4], hi=[0.6, 0.6])
        assert svg.index("<path") < svg.index("<polyline") < svg.index("<circle")

    def test_the_band_spans_p25_to_p75_not_the_full_axis(self):
        """A wide ribbon must mean a heterogeneous parcel, not a rendering default."""
        dates = ["2022-01-01", "2022-06-01"]
        narrow = trace_svg(dates, [0.5, 0.5], "2022-06-15",
                           lo=[0.48, 0.48], hi=[0.52, 0.52])
        wide = trace_svg(dates, [0.5, 0.5], "2022-06-15",
                         lo=[0.1, 0.1], hi=[0.9, 0.9])

        def band_height(svg):
            d = svg.split('<path d="M')[1].split('"')[0]
            ys = [float(p.split(",")[1]) for p in d.replace("L", " ").split()
                  if "," in p]
            return max(ys) - min(ys)

        assert band_height(narrow) < band_height(wide)

    def test_marks_outside_the_window_are_clipped_to_the_plot_area(self):
        """Y was always clamped; X never was.

        A stray circle outside the axes is a dot. A *filled* ribbon that escapes them is a
        smear across the whole card, so the data marks are clipped.
        """
        dates = ["2021-01-01", "2022-06-01", "2024-01-01"]   # first and last well outside
        svg = trace_svg(dates, [0.5] * 3, "2022-06-15",
                        lo=[0.4] * 3, hi=[0.6] * 3)
        assert '<clipPath id="tclip">' in svg
        body = svg.split('<g clip-path="url(#tclip)">')[1].split("</g>")[0]
        assert "<path" in body and "<polyline" in body and "<circle" in body
        # the dashed imagery-date rule is *outside* the clip: it is an axis annotation
        assert "stroke-dasharray" not in body

    def test_a_date_missing_its_quantiles_is_skipped_not_drawn_at_zero(self):
        dates = ["2022-01-01", "2022-03-01", "2022-06-01"]
        svg = trace_svg(dates, [0.5, 0.5, 0.5], "2022-06-15",
                        lo=[0.4, float("nan"), 0.4], hi=[0.6, float("nan"), 0.6])
        d = svg.split('<path d="M')[1].split('"')[0]
        assert d.count("L") == 4       # two usable dates, traced out along p75 and back
        assert svg.count("<circle") == 3   # the median point itself is still shown


class TestBuild:
    def test_shards_are_capped_at_the_requested_size(self):
        items = [{"item_id": f"M{i}"} for i in range(600)]
        parts = shard(items, 250)
        assert [len(p) for p in parts] == [250, 250, 100]

    def test_items_carry_only_allowlisted_fields(self, campaign):
        sample, chips, px, _ = campaign
        from crop_classifier.labelling.build_html import build_traces
        items = build_items(sample, chips, build_traces(px))
        extra = set(items[0]) - set(ITEM_FIELDS) - {"context", "zoom"}
        assert not extra, f"unexpected fields reached the labeller: {extra}"

    def test_no_shard_exceeds_16_mb(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        for f in out.glob("*.html"):
            assert f.stat().st_size <= 16 * 1024 * 1024, f.name

    def test_blindness__no_forbidden_value_appears_in_the_raw_html(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        for f in out.glob("*.html"):
            html = f.read_text()
            for field in FORBIDDEN_FIELDS:
                assert f'"{field}"' not in html, f"{field} leaked into {f.name}"
            # the *values* matter as much as the keys
            payload = html.split(ITEMS_OPEN)[1].split(ITEMS_CLOSE)[0]
            assert '"PERENNIAL"' not in payload
            assert "trainval" not in html
            assert "123.45" not in html

    def test_blindness_holds_on_the_OVERLAP_shard_too(self, campaign):
        """The blindness test above never built one — the fixture is `overlap=False`.

        So the guarantee had only ever been checked on solo shards. It holds here too, with
        one **known and intended** exception: the overlap shard is *named* `overlap`, so the
        string appears once as `const SHARD_ID="overlap"` and in the visible header. That is
        the file's own name, not the sample's `overlap` column, and it is on the page by
        design — both labellers are meant to receive this file.

        ⚠️ It does mean a labeller can tell which shard is double-labelled, and κ is
        measured on exactly that shard. Recorded rather than fixed: renaming would change
        the campaign structure (§D2) and the routing the filenames carry.
        """
        sample, chips, px, out = campaign
        sample = sample.copy()
        sample.loc[sample.index[:2], "overlap"] = True
        build(sample, chips, px, out)
        for f in out.glob("overlap_*.html"):
            html = f.read_text()
            for field in FORBIDDEN_FIELDS:
                if field == "overlap":
                    continue        # the shard id, see the docstring
                assert f'"{field}"' not in html, f"{field} leaked into {f.name}"
            payload = html.split(ITEMS_OPEN)[1].split(ITEMS_CLOSE)[0]
            assert '"PERENNIAL"' not in payload
            assert "trainval" not in html and "123.45" not in html
            # and the shard-id exception really is the only occurrence
            assert html.count('"overlap"') == 1

    def test_class_names_still_appear_as_the_labellers_buttons(self, campaign):
        """Blindness is about the declared class, not about the label vocabulary."""
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        for lab in LABELS:
            assert lab in html

    def test_unsure_is_offered_as_a_key_and_stays_on_5(self, campaign):
        """A control that must be actively set does not get used; a key does.

        `NON_AGRICULTURE` was added after the codebook was frozen. Renumbering the abstain
        to make the list read tidily would silently change what a briefed labeller's
        fingers do, so 5 is pinned here.
        """
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        assert "'5':'UNSURE'" in html
        assert LABELS.index("UNSURE") == 4 and len(LABELS) == 6

    def test_non_agriculture_is_offered_on_key_6(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        assert "'6':'NON_AGRICULTURE'" in html
        assert "NON_AGRICULTURE" in LABELS

    def test_the_codebook_states_the_other_vs_non_agriculture_boundary(self, campaign):
        """Two overlapping classes with no separating test is how kappa gets lost.

        `NON_AGRICULTURE` was carved out of `OTHER`, whose old definition explicitly listed
        water, built-up, road and riverbed. If the emitted page does not carry the rule
        that separates them, the two labellers are guessing at the same parcel.
        """
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        assert "sown next season" in html
        # ...and OTHER no longer claims the ground NON_AGRICULTURE now owns
        other = html.split("3 OTHER")[1].split("</li>")[0]
        for gone in ("riverbed", "built-up", "water"):
            assert gone not in other, f"OTHER still claims {gone!r}"


class TestSpanish:
    """The delivered set is Spanish; English stays as the reference build."""

    def test_every_language_defines_every_string_and_every_label(self):
        """A missing key renders as `undefined` on the page, never as an error."""
        keys = set(UI["en"])
        for lang in LANGS:
            assert set(UI[lang]) == keys, f"{lang} UI keys differ"
            assert set(LABEL_DISPLAY[lang]) == set(LABELS)
            assert lang in CODEBOOK_HTML

    def test_stored_label_values_stay_english_in_every_language(self, campaign):
        """Translating the *value* would make ingest match nothing and report no error.

        Only the display name is localised; the CSV and `ingest.LABELS` keep the canonical
        constants.
        """
        sample, chips, px, out = campaign
        build(sample, chips, px, out, lang="es")
        html = next(out.glob("*.html")).read_text()
        assert "'5':'UNSURE'" in html and "'6':'NON_AGRICULTURE'" in html
        assert "'label'" not in html.split("const KEYS=")[1][:200]

    def test_spanish_page_is_spanish(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out, lang="es")
        html = next(out.glob("*.html")).read_text()
        assert '<html lang="es">' in html
        assert "Etiquetado de parcelas" in html and "Descargar CSV" in html
        assert "PERENNE" in html and "NO AGR\u00cdCOLA" in html
        # the English interface must not leak through half-translated
        for stray in ("Download CSV", "your name", "Codebook &mdash; pick one"):
            assert stray not in html, f"untranslated: {stray!r}"

    def test_spanish_carries_the_same_decision_rules(self, campaign):
        """A shorter codebook is fine; a codebook missing a rule is not."""
        sample, chips, px, out = campaign
        build(sample, chips, px, out, lang="es")
        # whitespace-normalised: the source wraps these phrases across lines
        html = " ".join(next(out.glob("*.html")).read_text().split())
        assert "sembrar este suelo" in html          # the 3-vs-6 separating question
        assert "&gt;50 %" in html                    # the majority rule
        assert "Nunca ANUAL por defecto" in html     # young plantings
        assert "borde no coincide" in html           # boundary mismatch

    def test_both_codebooks_end_by_asking_for_the_csv(self):
        """The last thing a labeller reads has to be what to do with the file."""
        assert "Download CSV" in CODEBOOK_HTML["en"].split("When you finish")[1]
        assert "Descargar CSV" in CODEBOOK_HTML["es"].split("Al terminar")[1]

    def test_the_codebook_got_shorter(self):
        """It is read mid-task on a parcel that is already confusing."""
        for lang in LANGS:
            assert len(CODEBOOK_HTML[lang]) < 5200

    def test_an_unknown_language_is_refused_not_silently_english(self, campaign):
        sample, chips, px, out = campaign
        with pytest.raises(ValueError):
            build(sample, chips, px, out, lang="fr")
        with pytest.raises(ValueError):
            render_html([], "shard01", "A", lang="fr")


    def test_the_two_labellers_of_the_overlap_see_different_orders(self, campaign):
        """Same order would let fatigue line up between them and inflate kappa."""
        import json as _json
        sample, chips, px, out = campaign
        sample = sample.copy()
        sample["overlap"] = True
        build(sample, chips, px, out)
        def ids(f):
            t = (out / f).read_text()
            return [x["item_id"]
                    for x in _json.loads(t.split(ITEMS_OPEN)[1].split(ITEMS_CLOSE)[0])]
        a, b = ids("overlap_A.html"), ids("overlap_B.html")
        assert sorted(a) == sorted(b) and a != b

    def test_shard_order_is_reproducible_across_builds(self, campaign):
        """`hash()` is salted per run; two builds must be diffable."""
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        first = (out / "shard01_A.html").read_text()
        build(sample, chips, px, out)
        assert (out / "shard01_A.html").read_text() == first

    def test_every_item_id_round_trips_through_the_key(self, campaign):
        sample, chips, px, out = campaign
        key = build(sample, chips, px, out)
        assert set(key.item_id) == set(sample.item_id)
        assert key.COD_PREDIO.notna().all()
        assert key.groupby("item_id")["COD_PREDIO"].nunique().max() == 1

    def test_overlap_parcels_go_to_both_labellers(self, campaign):
        sample, chips, px, out = campaign
        sample = sample.copy()
        sample.loc[sample.index[:2], "overlap"] = True
        build(sample, chips, px, out)
        names = sorted(f.name for f in out.glob("overlap_*.html"))
        assert names == ["overlap_A.html", "overlap_B.html"]

    def test_embedded_json_parses(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        payload = html.split(ITEMS_OPEN)[1].split(ITEMS_CLOSE)[0]
        items = json.loads(payload)
        assert items and all("context" in it and "zoom" in it for it in items)
        # the base64 must actually decode to an image the browser can show
        Image.open(io.BytesIO(base64.b64decode(items[0]["context"])))


class TestNoConfidenceControl:
    """Confidence was a second, softer abstain competing with UNSURE."""

    def test_no_confidence_column_in_the_downloaded_csv(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        header = html.split("const head=[")[1].split("]")[0]
        assert "confidence" not in header
        assert "'label'" in header

    def test_the_c_key_and_its_control_are_gone(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = next(out.glob("*.html")).read_text()
        assert "cycleConf" not in html
        assert "e.key==='c'" not in html
        assert 'id="conf"' not in html

class TestAnnotatorName:
    """The annotator types their own name; the filename letter is only a suggestion."""

    def test_the_name_box_is_present_and_the_csv_uses_it(self, campaign):
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = (out / "shard01_A.html").read_text()
        assert 'id="who"' in html
        # the CSV row is written from the typed name, not from the filename constant
        assert "q(who)" in html
        assert "SHARD_ID+'_'+who" in html

    def test_progress_is_keyed_by_shard_and_name(self, campaign):
        """Two people sharing a browser must not overwrite each other."""
        sample, chips, px, out = campaign
        build(sample, chips, px, out)
        html = (out / "shard01_A.html").read_text()
        assert "'s2label_'+SHARD_ID+'_'+(who||'anon')" in html

    def test_the_key_records_suggested_labellers_not_assignments(self, campaign):
        sample, chips, px, out = campaign
        key = build(sample, chips, px, out)
        assert "suggested_labellers" in key.columns


class TestRenderHtml:
    def test_is_self_contained(self):
        html = render_html([{"item_id": "M1", "dept": "Piura", "imagery_date": "2022-06-15",
                             "imagery_res": "1.2m", "area_ha": 1.0, "n_obs": 3,
                             "detail": "", "context": "", "trace": "<svg></svg>"}],
                           "shard01", "A")
        assert "<script src=" not in html and "<link" not in html
        assert "http://" not in html and "https://" not in html.replace(
            "http://www.w3.org/2000/svg", "")

    def test_download_works_from_disk_not_only_as_an_artifact(self):
        html = render_html([], "shard01", "A")
        assert "createObjectURL" in html and "window.claude" in html


class TestAttrition:
    """A shard quietly 40 parcels short is a 4 % cut nobody would notice until the gates."""

    def test_a_parcel_with_no_chip_is_reported_not_silently_dropped(self, campaign):
        sample, chips, px, out = campaign
        (chips / "M00000_context.jpg").unlink()
        key = build(sample, chips, px, out)
        assert "M00000" not in set(key.item_id)
        rep = pd.read_csv(out / "excluded_from_shards.csv")
        assert (rep.item_id == "M00000").any()
        assert "no chip" in rep.loc[rep.item_id == "M00000", "reason"].iloc[0]

    def test_a_parcel_with_no_s2_trace_is_kept_and_flagged(self, campaign):
        sample, chips, px, out = campaign
        px = px[px.COD_PREDIO != "P0"]
        key = build(sample, chips, px, out)
        assert "M00000" in set(key.item_id)          # still labellable from imagery
        rep = pd.read_csv(out / "excluded_from_shards.csv")
        assert "no S2 trace" in rep.loc[rep.item_id == "M00000", "reason"].iloc[0]
