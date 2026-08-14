"""Unit tests for the labelling chip panels.

The panel geometry had no coverage until the left panel became a *context* view. What is
pinned here is the thing that would fail silently: a context panel narrower than the zoom
panel it is supposed to give context for, or a neighbour lookup that does not reach as far
as the panel it feeds. Neither raises anything — both just render a picture that quietly
misleads the labeller.
"""

from __future__ import annotations

import numpy as np

from crop_classifier.labelling.chips import (
    CONTEXT_MIN_M,
    CONTEXT_PAD_FRAC,
    CONTEXT_PAD_MAX_M,
    ZOOM_M,
    _nice_bar,
    context_extent_m,
    is_placeholder,
)


class TestContextExtent:
    def test_a_small_parcel_gets_the_floor_not_its_own_tiny_extent(self):
        """The 5th-percentile parcel is 66 m across; edge-to-edge shows no setting at all."""
        assert context_extent_m(66.0) == CONTEXT_MIN_M

    def test_padding_is_proportional_for_a_mid_sized_parcel(self):
        """A 600 m parcel: 300 m of margin each side, so the panel is 2x the parcel."""
        assert context_extent_m(600.0) == 600.0 + 2 * CONTEXT_PAD_MAX_M

    def test_padding_is_a_fraction_before_it_hits_the_cap(self):
        span = 300.0                      # 0.5 * 300 = 150 < 300 cap
        assert context_extent_m(span) == span * (1 + 2 * CONTEXT_PAD_FRAC)

    def test_the_cap_keeps_the_largest_parcel_from_demanding_a_4_km_mosaic(self):
        """The largest parcel in the draw is 2,127 m across."""
        assert context_extent_m(2127.0) == 2127.0 + 2 * CONTEXT_PAD_MAX_M
        assert context_extent_m(2127.0) < 4000.0

    def test_the_whole_parcel_always_fits(self):
        """The codebook promises the whole parcel outlined; padding must never crop it."""
        for span in (10, 66, 157, 265, 627, 1208, 2127):
            assert context_extent_m(float(span)) >= span

    def test_context_is_always_wider_than_the_zoom_panel(self):
        """Two panels where the 'context' is the narrower one is worse than one panel.

        This is the constraint that binds when ZOOM_M is raised: at ZOOM_M = 200 a floor of
        120 m — the value the old detail panel used — would invert the pair for every
        parcel below ~150 m, which is half the draw.
        """
        for span in (10, 66, 157, 265, 627, 1208, 2127):
            assert context_extent_m(float(span)) > ZOOM_M

    def test_it_is_monotone_in_parcel_size(self):
        spans = np.array([10, 50, 100, 200, 400, 800, 1600, 3200], dtype=float)
        ext = np.array([context_extent_m(s) for s in spans])
        assert (np.diff(ext) >= 0).all()


class TestScaleBar:
    def test_the_bar_is_about_a_quarter_of_the_panel(self):
        """A bar spanning the panel reads as a border, not a scale."""
        for extent in (200.0, 400.0, 530.0, 1227.0, 2727.0):
            bar, _ = _nice_bar(extent)
            assert 0.2 <= bar / extent <= 0.55

    def test_the_two_panels_get_different_bars_so_they_are_not_read_as_one_scale(self):
        assert _nice_bar(ZOOM_M) != _nice_bar(context_extent_m(600.0))


class TestPlaceholderDetection:
    def test_esris_flat_grey_not_available_tile_is_caught(self):
        """It arrives as a *valid image*, not an error — the original silent failure."""
        grey = np.full((64, 64, 3), 205, dtype=np.uint8)
        assert is_placeholder(grey)

    def test_real_imagery_is_not_mistaken_for_it(self):
        rng = np.random.default_rng(0)
        scene = rng.integers(20, 200, size=(64, 64, 3)).astype(np.uint8)
        assert not is_placeholder(scene)
