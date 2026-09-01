"""Closed routes: code whose **estimand** was abandoned after a pre-registered gate failed.

Nothing here is broken. Each module is built, unit-tested and correct; what failed was the
question it answers, measured, and written down. It stays in the tree because "we tried this
and here is why it does not work" is a result — the alternative is that the next person spends
a month rediscovering it.

It lives in its own package, and under ``cc archive``, so that browsing ``src/`` or running
``cc --help`` shows the pipeline someone would actually run rather than 45 commands of which
seven are dead ends.

See ``README.md`` beside this file for what each module tried and which gate killed it, and
``docs/RESULTS.md`` §5, §6.4, §7 and §9 for the numbers.

⚠️ **Not everything closed is in here.** ``perennial/panel.py``, ``diagnostics.py``,
``trajectories.py`` and ``harmonization.py`` belong to failed estimands too, but live code
depends on them — ``features/assemble.py`` calls ``harmonization.oli_to_etm``, and
``perennial/report_figures.py`` builds committed figures out of the panel. Moving them would
tangle the live path to tidy the dead one, so they stay where they are and are marked in
``docs/RESULTS.md`` instead.
"""
