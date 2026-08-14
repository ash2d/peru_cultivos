"""Perennial vs annual vs pasture/fallow: the 3-class land-state classifier.

See ``docs/perennial/plan.md``. This package holds only what is *new* relative to the
12-class pipeline — the label build, the rule-based model, the multi-year panel, the
MapBiomas benchmark and the trajectory analysis. Everything else (splits, GEE extraction,
feature assembly, training, evaluation) is reused unchanged via the ``CC_PROC`` workspace
switch in ``crop_classifier.paths``.
"""
