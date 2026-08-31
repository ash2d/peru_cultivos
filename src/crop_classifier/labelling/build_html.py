"""Build the self-contained labelling HTML shards (docs/s2_labelling/plan.md).

One file per shard of 200 parcels: two base64 JPEGs + an inline SVG NDVI trace per item
(~58 KB), so a shard is ~12 MB — openable from disk with no server.

**The annotator types their own name.** The letter in the filename (``shard01_A.html``) is
only a suggestion of who should take that shard; whatever is typed into the name box is
what lands in the CSV and in the download filename. Progress in ``localStorage`` is keyed
by shard *and* name, so two people can share a browser without overwriting each other.

**Blindness is a guarantee, not an intention.** The declared PETT class, the train/test
assignment and the fold must not appear *anywhere* in the emitted file, not merely be
un-rendered: a labeller anchored on the 1998 declaration manufactures agreement between
declaration and endpoint, which is the human form of the ``centroid_lat`` failure. The
embedded JSON therefore carries an explicit allow-list of fields and
``tests/test_build_html.py`` asserts on the raw string.

Run::

    CC_PROC=data/processed/all_peru uv run python -m crop_classifier.cli \\
        allperu label-html
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 200, not the plan's 250. Measured on real chips: two base64 JPEGs plus a ~6 KB SVG trace
# come to ~58 KB per item, so 250 items is ~14.5 MB against a 16 MB cap — inside it, but
# with no room for the handful of parcels whose chips compress badly. 200 lands at ~11.6 MB.
SHARD_SIZE = 200
# Everything the labeller may see. Anything not on this list never reaches the file.
ITEM_FIELDS = ["item_id", "dept", "imagery_date", "imagery_res", "area_ha",
               "n_obs", "trace"]
# Six values: five real classes and an abstain.
#
# `UNSURE` is an explicit abstain. The plan originally forced a choice and made
# `confidence = 1` the abstain signal, on the reasoning that a forced label plus a
# confidence flag loses nothing. In practice a control that has to be *actively* set is not
# used — the labeller picks a class and moves on — so the abstain silently never got
# recorded and parcels that could not really be called were indistinguishable from ones
# that could. A key that costs the same as any other class key is the only version that
# gets pressed. **Confidence has since been removed entirely**: it was a second, softer
# abstain competing with the first, and two ways to say "I am not sure" split the signal.
#
# ⚠️ **Order is key order, and `UNSURE` deliberately stays on 5.** `NON_AGRICULTURE` was
# added after the codebook was written, and renumbering the abstain to make the list read
# tidily would silently change what an already-briefed labeller's fingers do.
LABELS = ["PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP", "UNSURE",
          "NON_AGRICULTURE"]

# ⚠️ Translation touches the **display name only**. The value written to the CSV, and every
# value `ingest.py` compares against, stays the canonical English constant above. Localising
# the stored value would mean the ingest silently matched nothing and reported an empty
# label distribution rather than an error — the same class of failure as the four data traps.
LABEL_DISPLAY = {
    "en": {lab: lab for lab in LABELS},
    "es": {"PERENNIAL": "PERENNE", "ANNUAL": "ANUAL", "OTHER": "OTRO",
           "WOODY_NON_CROP": "LEÑOSO NO CULTIVO", "UNSURE": "NO SEGURO",
           "NON_AGRICULTURE": "NO AGRÍCOLA"},
}
LANGS = tuple(LABEL_DISPLAY)

# Fields that must NEVER appear, checked by the test suite against the raw HTML.
FORBIDDEN_FIELDS = ["declared_class", "label_id", "split", "fold", "crop_set",
                    "stratum", "weight", "region_id", "N_h", "n_h", "batch", "overlap"]

SVG_W, SVG_H = 340, 116
NDVI_LO, NDVI_HI = -0.1, 1.0     # fixed y-axis so parcels are comparable across the set


# ------------------------------------------------------------------------------------
# NDVI trace -> inline SVG
# ------------------------------------------------------------------------------------
def trace_svg(dates: list[str], ndvi: list[float], centre: str,
              w: int = SVG_W, h: int = SVG_H,
              lo: list[float] | None = None,
              hi: list[float] | None = None) -> str:
    """24-month NDVI trace: points on a light line, a vertical rule at ``centre``.

    Inline SVG rather than a rendered PNG — sharper, about a third of the size, and
    readable at any zoom. The y-axis is **fixed** at -0.1 to 1.0 so a flat fallow parcel
    and a peaking annual are visually comparable across the whole set; an autoscaled axis
    would make every parcel look like it has a season.

    ``lo``/``hi`` are the p25/p75 of NDVI **across the parcel's pixels on each date**, and
    when given they are drawn as a shaded ribbon behind the line. That spread is a real
    discriminator the median alone hides: a uniform annual field is internally consistent
    and reads as a narrow ribbon, whereas an orchard (crowns against bare inter-row) and a
    part-converted or mixed parcel are wide. It is **not** a confidence interval on the
    median and the codebook says so — a wide ribbon means a heterogeneous parcel, not a
    poorly measured one.

    Both are optional: a store without the quantile columns still renders the plain median
    line, so the page degrades rather than failing.
    """
    pad_l, pad_r, pad_t, pad_b = 26, 6, 8, 16
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    t = pd.to_datetime(pd.Series(dates))
    c = pd.Timestamp(centre)
    # `t_lo`/`t_hi`, not `lo`/`hi`: those names are the p25/p75 arguments. The first
    # version of the ribbon reused them here and the band was silently drawn from
    # timestamps.
    t_lo = c - pd.DateOffset(months=12)
    t_hi = c + pd.DateOffset(months=12)
    span = (t_hi - t_lo).days or 1

    def X(ts):
        return pad_l + iw * float((pd.Timestamp(ts) - t_lo).days) / span

    def Y(v):
        v = min(max(float(v), NDVI_LO), NDVI_HI)
        return pad_t + ih * (1 - (v - NDVI_LO) / (NDVI_HI - NDVI_LO))

    parts = [f'<svg class="trace" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
             f'xmlns="http://www.w3.org/2000/svg">']
    # Clip the data marks to the plot area. The extraction window and `centre` agree by
    # construction, so nothing *should* fall outside — but Y is clamped and X never was, and
    # a filled ribbon that escapes the axes is a smear across the whole card, where a stray
    # circle was merely a dot. Cheap insurance against a mismatch nobody would look for.
    # One trace is in the DOM at a time (the page injects the current item), so a fixed id
    # cannot collide.
    parts.append(f'<defs><clipPath id="tclip"><rect x="{pad_l}" y="{pad_t}" '
                 f'width="{iw}" height="{ih}"/></clipPath></defs>')
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" '
                 f'fill="#0d1117" stroke="#2b3440"/>')
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(gv)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw}" y2="{y:.1f}" '
                     f'stroke="#232c36"/>')
        parts.append(f'<text x="{pad_l - 4}" y="{y + 3:.1f}" text-anchor="end" '
                     f'font-size="8" fill="#7d8996">{gv:g}</text>')
    # month-boundary ticks, labelled every 6 months
    m = t_lo.to_period("M").to_timestamp()
    while m <= t_hi:
        if m >= t_lo:
            x = X(m)
            major = m.month in (1, 7)
            parts.append(f'<line x1="{x:.1f}" y1="{pad_t + ih}" x2="{x:.1f}" '
                         f'y2="{pad_t + ih + (4 if major else 2)}" stroke="#41505f"/>')
            if major:
                parts.append(f'<text x="{x:.1f}" y="{h - 3}" text-anchor="middle" '
                             f'font-size="8" fill="#7d8996">{m:%b %y}</text>')
        m += pd.DateOffset(months=1)

    parts.append('<g clip-path="url(#tclip)">')
    # The ribbon is drawn FIRST so the median line and its points sit on top of it. A band
    # painted over the line would hide exactly the series it is meant to qualify.
    if lo is not None and hi is not None:
        band = [(X(ts), Y(a), Y(b)) for ts, v, a, b in zip(t, ndvi, lo, hi)
                if pd.notna(v) and pd.notna(a) and pd.notna(b)]
        if len(band) > 1:
            up = " ".join(f"L{x:.1f},{yh:.1f}" for x, _, yh in band)
            dn = " ".join(f"L{x:.1f},{yl:.1f}" for x, yl, _ in reversed(band))
            parts.append(f'<path d="M{band[0][0]:.1f},{band[0][2]:.1f} {up} {dn} Z" '
                         f'fill="#4ade80" fill-opacity="0.20" stroke="none"/>')

    pts = [(X(ts), Y(v)) for ts, v in zip(t, ndvi) if pd.notna(v)]
    if len(pts) > 1:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="#3f6f8f" '
                     f'stroke-width="1"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7" fill="#4ade80"/>')
    parts.append("</g>")
    xc = X(c)
    parts.append(f'<line x1="{xc:.1f}" y1="{pad_t}" x2="{xc:.1f}" y2="{pad_t + ih}" '
                 f'stroke="#ffdd33" stroke-width="1.2" stroke-dasharray="3,2"/>')
    parts.append("</svg>")
    return "".join(parts)


def build_traces(px: pd.DataFrame) -> dict[str, dict]:
    """``COD_PREDIO -> {dates, ndvi, lo, hi}`` from the per-date S2 store.

    **The line and the ribbon must be the same quantity.** When the store carries the
    per-pixel NDVI quartiles (``NDVI_px_p25/p50/p75``, see ``s2_gee.S2_NDVI_BAND``) the
    line is the per-pixel *p50*, not the NDVI recomputed from the band medians. The two
    differ — a quantile of a ratio is not the ratio of the quantiles, measured at ~0.0025
    NDVI median absolute difference on this store — and mixing them would let the plotted
    median sit visibly outside its own p25-p75 band on some dates, which reads as a bug.

    A store without those columns (anything extracted before 2026-08-13) falls back to the
    band-median NDVI and no ribbon.
    """
    from crop_classifier.features.indices import add_indices, scale_sr_s2
    from crop_classifier.features.s2_gee import NDVI_PX_COLS

    d = add_indices(scale_sr_s2(px))
    d = d.sort_values(["COD_PREDIO", "date"])
    has_q = all(c in d.columns for c in NDVI_PX_COLS)
    p25, p50, p75 = NDVI_PX_COLS
    out = {}
    for cid, g in d.groupby("COD_PREDIO"):
        rec = {"dates": [str(x)[:10] for x in g["date"]],
               "ndvi": [round(float(v), 3)
                        for v in (g[p50] if has_q else g["NDVI"])]}
        if has_q:
            rec["lo"] = [round(float(v), 3) for v in g[p25]]
            rec["hi"] = [round(float(v), 3) for v in g[p75]]
        out[str(cid)] = rec
    return out


# ------------------------------------------------------------------------------------
# Items
# ------------------------------------------------------------------------------------
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_items(sample: pd.DataFrame, chip_dir: Path, traces: dict[str, dict],
                dropped: list[dict] | None = None) -> list[dict]:
    """One embedded record per parcel — only :data:`ITEM_FIELDS`, nothing else.

    A parcel with no rendered chip cannot be labelled and is left out, but it is
    **appended to ``dropped``** rather than vanishing: a shard that is quietly 40 parcels
    short is a 4 % cut to the campaign that nobody would notice until the gates ran.
    Parcels with a chip but *no S2 trace* are kept — the labeller can still call them from
    the imagery, and the empty trace is visible on the page.
    """
    items = []
    for _, r in sample.iterrows():
        det = chip_dir / f"{r.item_id}_context.jpg"
        zoom = chip_dir / f"{r.item_id}_zoom.jpg"
        if not (det.exists() and zoom.exists()):
            if dropped is not None:
                dropped.append({"item_id": r.item_id, "COD_PREDIO": r.COD_PREDIO,
                                "dept": r.dept, "reason": "no chip rendered"})
            continue
        tr = traces.get(str(r.COD_PREDIO), {"dates": [], "ndvi": []})
        if not tr["ndvi"] and dropped is not None:
            dropped.append({"item_id": r.item_id, "COD_PREDIO": r.COD_PREDIO,
                            "dept": r.dept, "reason": "no S2 trace (kept, imagery only)"})
        items.append({
            "item_id": str(r.item_id),
            "dept": str(r.dept).replace("_", " ").title(),
            "imagery_date": str(r.imagery_date)[:10],
            "imagery_res": str(r.imagery_res),
            "area_ha": round(float(r.area_ha), 2),
            "n_obs": len(tr["ndvi"]),
            "context": _b64(det),
            "zoom": _b64(zoom),
            "trace": trace_svg(tr["dates"], tr["ndvi"], str(r.imagery_date)[:10],
                               lo=tr.get("lo"), hi=tr.get("hi")),
        })
    return items


def shard(items: list[dict], size: int = SHARD_SIZE) -> list[list[dict]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _stable_seed(text: str) -> int:
    """Deterministic 32-bit seed from a string.

    ``hash()`` is randomised per interpreter run (PYTHONHASHSEED), so seeding a shuffle
    with it means rebuilding the shards reshuffles them — the same parcels in a different
    order, which is harmless but makes two builds impossible to diff.
    """
    import hashlib
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------------------------
# HTML
# ------------------------------------------------------------------------------------
_CSS = """
:root{--bg:#0b0f14;--panel:#121821;--ink:#e6edf3;--dim:#8b98a5;--line:#232c36;
      --acc:#ffdd33;--ok:#4ade80}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--panel);
       border-bottom:1px solid var(--line);padding:8px 14px;display:flex;
       gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.02em}
.bar{flex:1;min-width:160px;height:6px;background:#1c242e;border-radius:3px;
     overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ok);width:0}
button{background:#1c242e;color:var(--ink);border:1px solid var(--line);
       border-radius:5px;padding:5px 10px;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}
#wrap{max-width:1180px;margin:0 auto;padding:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
      padding:12px;display:grid;grid-template-columns:auto auto 1fr;gap:14px;
      align-items:start}
.card img{width:320px;height:320px;border-radius:4px;background:#000;display:block}
.cap{font-size:11px;color:var(--dim);margin-top:3px;text-align:center}
.meta{font-size:12px;color:var(--dim);margin-bottom:8px}
.meta b{color:var(--ink);font-weight:600}
.keys{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 6px}
.key{border:1px solid var(--line);border-radius:5px;padding:6px 9px;font-size:12px;
     cursor:pointer;background:#0f151d}
.key.sel{background:var(--acc);color:#111;border-color:var(--acc);font-weight:600}
.key i{color:var(--dim);font-style:normal;margin-right:5px}
.key.sel i{color:#111}
.row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
input[type=text]{background:#0f151d;border:1px solid var(--line);border-radius:5px;
                 color:var(--ink);padding:5px 8px;font-size:12px;width:190px}
label.cb{font-size:12px;color:var(--dim);display:flex;gap:5px;align-items:center;
         cursor:pointer}
.trace{background:#0d1117;border-radius:4px}
.note{font-size:11px;color:var(--dim);margin-top:8px;line-height:1.5}
#banner{background:#2b2410;border:1px solid #6b5a15;color:#ffdd33;padding:8px 12px;
        border-radius:6px;margin-bottom:12px;font-size:12px;display:none}
#done{color:var(--ok);font-weight:600}
kbd{background:#0f151d;border:1px solid var(--line);border-radius:3px;padding:1px 5px;
    font-size:11px}
@media (max-width:1000px){.card{grid-template-columns:1fr}.card img{width:100%;height:auto}}
"""

# ------------------------------------------------------------------------------------
# User-visible strings
#
# Every user-visible string lives here and is injected as `const T` next to the payload;
# the JS below only ever reads `T.*` and `NAMES[...]`. One code path serves both languages,
# so a fix to the labelling logic cannot land in one language and miss the other.
# ------------------------------------------------------------------------------------
UI = {
    "en": {
        "title": "Parcel labelling", "your_name": "your name",
        "name_ph": "type your name", "download": "Download CSV", "codebook": "Codebook",
        "labelled": "labelled",
        "cap_context": "context &mdash; whole parcel + surroundings, "
                       "outlined yellow, neighbours cyan",
        "cap_zoom": "zoom &mdash; 200 m across, same centre",
        "imagery": "imagery",
        "cap_trace_a": "Sentinel-2 NDVI, 24 months &mdash; ",
        "cap_trace_b": " clear observations; line = parcel median, shaded band = the "
                       "middle half of its pixels (p25&ndash;p75); dashed line = the "
                       "imagery date",
        "bm": "boundary no longer matches the visible field",
        "crop_ph": "crop guess (optional)",
        "hint_tail": "&nbsp;·&nbsp; <kbd>b</kbd> boundary &nbsp;"
                     "<kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate. Typing in a text box "
                     "disables the shortcuts until you click away.",
        "need_name": "Please type your name in the box at the top first — it goes into "
                     "the file so we know whose labels these are.",
        "resumed_a": "Resumed &mdash; ", "resumed_b": " of ",
        "resumed_c": " labels restored from this browser. ",
        "go_first": "go to the first unlabelled",
    },
    "es": {
        "title": "Etiquetado de parcelas", "your_name": "su nombre",
        "name_ph": "escriba su nombre", "download": "Descargar CSV", "codebook": "Manual",
        "labelled": "etiquetadas",
        "cap_context": "contexto &mdash; parcela completa y su entorno; "
                       "contorno amarillo, vecinas en celeste",
        "cap_zoom": "acercamiento &mdash; 200 m de ancho, mismo centro",
        "imagery": "imagen",
        "cap_trace_a": "NDVI Sentinel-2, 24 meses &mdash; ",
        "cap_trace_b": " observaciones despejadas; línea = mediana de la parcela, banda "
                       "sombreada = la mitad central de sus píxeles (p25&ndash;p75); "
                       "línea punteada = fecha de la imagen",
        "bm": "el borde ya no coincide con la chacra visible",
        "crop_ph": "cultivo probable (opcional)",
        "hint_tail": "&nbsp;·&nbsp; <kbd>b</kbd> borde &nbsp;"
                     "<kbd>&larr;</kbd><kbd>&rarr;</kbd> navegar. Mientras escribe en un "
                     "cuadro de texto los atajos quedan desactivados.",
        "need_name": "Escriba su nombre en el cuadro de arriba antes de descargar — va "
                     "dentro del archivo para saber de quién son las etiquetas.",
        "resumed_a": "Reanudado &mdash; ", "resumed_b": " de ",
        "resumed_c": " etiquetas recuperadas de este navegador. ",
        "go_first": "ir a la primera sin etiquetar",
    },
}

_KEY_HINT = {
    "en": ["perennial", "annual", "other", "woody non-crop", "unsure", "non-agriculture"],
    "es": ["perenne", "anual", "otro", "leñoso no cultivo", "no seguro", "no agrícola"],
}


def _js_head(lang: str) -> str:
    """`T` (interface strings) and `NAMES` (button captions) for one language."""
    names = LABEL_DISPLAY[lang]
    hint = " &nbsp;".join(f"<kbd>{i + 1}</kbd> {w}"
                          for i, w in enumerate(_KEY_HINT[lang]))
    t = dict(UI[lang], hint=hint + " " + UI[lang]["hint_tail"])
    return (f"const T={json.dumps(t, ensure_ascii=False)};\n"
            f"const NAMES={json.dumps(names, ensure_ascii=False)};\n")


_JS = r"""
const KEYS={'1':'PERENNIAL','2':'ANNUAL','3':'OTHER','4':'WOODY_NON_CROP','5':'UNSURE',
            '6':'NON_AGRICULTURE'};
const NAMEK='s2label_annotator';
let i=0, store={}, t0=Date.now(), who='';

// The annotator types their own name; the filename's letter is only a suggestion of who
// should do this shard. The name is stored globally (one key, not per shard) so it is
// typed once even when someone works through several files, and the per-shard progress
// store is keyed BY that name so two people sharing a browser never overwrite each other.
function shardKey(){ return 's2label_'+SHARD_ID+'_'+(who||'anon'); }
function loadStore(){ try{store=JSON.parse(localStorage.getItem(shardKey())||'{}')}
  catch(e){store={}} }
// No `confidence` field. It was a second, softer abstain sitting beside UNSURE, and two
// ways to say "I am not sure" split that signal between a hard one the gates can read and
// a graded one nobody moved off its default.
function rec(id){ return store[id] || (store[id]={label:'',crop_guess:'',
  boundary_mismatch:false,seconds_spent:0}); }
function save(){ try{localStorage.setItem(shardKey(),JSON.stringify(store))}catch(e){} }

function setWho(v){
  const old=shardKey(), had=Object.keys(store).length;
  who=(v||'').trim();
  const nw=shardKey();
  if(nw!==old&&had&&!localStorage.getItem(nw)){
    // they started labelling before typing a name — carry that work over rather than
    // stranding it under the anonymous key
    try{localStorage.setItem(nw,JSON.stringify(store));localStorage.removeItem(old);}catch(e){}
  } else if(nw!==old){ loadStore(); }
  try{localStorage.setItem(NAMEK,who)}catch(e){}
  const w=document.getElementById('who'); if(w&&w.value!==who) w.value=who;
  render();
}
function nDone(){ return ITEMS.filter(x=>store[x.item_id]&&store[x.item_id].label).length }

function render(){
  const it=ITEMS[i], r=rec(it.item_id);
  document.getElementById('pos').textContent=(i+1)+' / '+ITEMS.length;
  document.getElementById('cnt').textContent=nDone()+' '+T.labelled;
  document.getElementById('pb').style.width=(100*nDone()/ITEMS.length)+'%';
  document.getElementById('wrap').innerHTML=
   '<div class="card">'
   +'<div><img src="data:image/jpeg;base64,'+it.context+'"><div class="cap">'
   +T.cap_context+'</div></div>'
   +'<div><img src="data:image/jpeg;base64,'+it.zoom+'"><div class="cap">'
   +T.cap_zoom+'</div></div>'
   +'<div><div class="meta"><b>'+it.item_id+'</b> &nbsp;·&nbsp; '+it.dept
   +' &nbsp;·&nbsp; '+it.area_ha+' ha &nbsp;·&nbsp; '+T.imagery+' <b>'+it.imagery_date
   +'</b> ('+it.imagery_res+')</div>'
   +it.trace
   +'<div class="cap">'+T.cap_trace_a+it.n_obs+T.cap_trace_b+'</div>'
   +'<div class="keys" id="keys"></div>'
   +'<div class="row">'
   +'<label class="cb"><input type="checkbox" id="bm"> '+T.bm+'</label></div>'
   +'<div class="row"><input type="text" id="crop" placeholder="'+T.crop_ph+'">'
   +'</div>'
   +'<div class="note">'+T.hint+'</div></div></div>';
  const kk=document.getElementById('keys');
  Object.keys(KEYS).forEach(function(k){
    const b=document.createElement('div');
    b.className='key'+(r.label===KEYS[k]?' sel':'');
    b.innerHTML='<i>'+k+'</i>'+NAMES[KEYS[k]];
    b.onclick=function(){setLabel(KEYS[k])};
    kk.appendChild(b);
  });
  document.getElementById('bm').checked=!!r.boundary_mismatch;
  document.getElementById('bm').onchange=function(e){rec(it.item_id).boundary_mismatch=
    e.target.checked;save()};
  const ci=document.getElementById('crop');
  ci.value=r.crop_guess||'';
  ci.oninput=function(e){rec(it.item_id).crop_guess=e.target.value;save()};
  t0=Date.now();
}
function tick(){ const r=rec(ITEMS[i].item_id);
  r.seconds_spent=Math.round((r.seconds_spent||0)+(Date.now()-t0)/1000); t0=Date.now(); }
function setLabel(v){ const r=rec(ITEMS[i].item_id); tick(); r.label=v;
  r.timestamp=new Date().toISOString(); save(); go(1); }
function go(d){ tick(); i=Math.min(ITEMS.length-1,Math.max(0,i+d)); render(); }

document.addEventListener('keydown',function(e){
  if(e.target.tagName==='INPUT') return;
  if(KEYS[e.key]) {setLabel(KEYS[e.key]); e.preventDefault(); return;}
  if(e.key==='b'){const r=rec(ITEMS[i].item_id);
    r.boundary_mismatch=!r.boundary_mismatch;save();render();e.preventDefault();}
  if(e.key==='ArrowLeft'){go(-1);e.preventDefault();}
  if(e.key==='ArrowRight'){go(1);e.preventDefault();}
});

function csv(){
  tick(); save();
  if(!who){ alert(T.need_name);
            const w=document.getElementById('who'); if(w) w.focus(); return; }
  const head=['item_id','labeller','label','crop_guess',
              'boundary_mismatch','seconds_spent','timestamp'];
  const q=function(s){s=(s==null?'':String(s));
    return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
  const lines=[head.join(',')];
  ITEMS.forEach(function(it){ const r=store[it.item_id]; if(!r||!r.label) return;
    lines.push([it.item_id,q(who),r.label,q(r.crop_guess),
                r.boundary_mismatch?'True':'False',r.seconds_spent||0,
                r.timestamp||''].join(','));});
  const text=lines.join('\n')+'\n';
  const name=SHARD_ID+'_'+who.replace(/[^A-Za-z0-9_-]+/g,'_')+'.csv';
  // Plain Blob + <a download> so the file works with the page opened straight from disk;
  // window.claude.downloads.save is used only when the page is running as an artifact.
  if(window.claude&&window.claude.downloads&&window.claude.downloads.save){
    try{ window.claude.downloads.save(name,text); return; }catch(e){}
  }
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'}));
  a.download=name; a.click(); URL.revokeObjectURL(a.href);
}

window.addEventListener('load',function(){
  try{who=localStorage.getItem(NAMEK)||''}catch(e){who=''}
  const w=document.getElementById('who');
  if(w){ w.value=who; w.onchange=function(e){setWho(e.target.value)};
         w.onblur=function(e){setWho(e.target.value)}; }
  loadStore();
  if(!who&&w){ w.focus(); }
  if(nDone()>0){ const b=document.getElementById('banner');
    b.style.display='block';
    b.innerHTML=T.resumed_a+nDone()+T.resumed_b+ITEMS.length+T.resumed_c+
      '<button onclick="jumpNext()">'+T.go_first+'</button>';
  }
  render();
});
function jumpNext(){ for(let k=0;k<ITEMS.length;k++){
    if(!store[ITEMS[k].item_id]||!store[ITEMS[k].item_id].label){i=k;break;} }
  render(); }
window.addEventListener('beforeunload',function(){tick();save();});
"""


ITEMS_OPEN = "const ITEMS="
ITEMS_CLOSE = ";\n/*end-items*/\n"


def render_html(items: list[dict], shard_id: str, labeller: str,
                codebook_html: str | None = None, lang: str = "en") -> str:
    """One self-contained shard file, in ``lang``.

    Written as flat concatenation rather than an indented triple-quoted template: the CSS
    and JS blocks interpolated into it contain unindented lines, so ``textwrap.dedent``
    silently does nothing and the emitted file's structure stops matching what the code
    reads like. ``ITEMS_OPEN``/``ITEMS_CLOSE`` are exact delimiters so the embedded
    payload can be recovered byte-for-byte by a test.
    """
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    u = UI[lang]
    if codebook_html is None:
        codebook_html = CODEBOOK_HTML[lang]
    payload = json.dumps(items, separators=(",", ":"))
    return (
        f'<!doctype html>\n<html lang="{lang}"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{u['title']} — {shard_id} ({labeller})</title>\n"
        f"<style>{_CSS}</style></head><body>\n"
        '<header>\n'
        f"<h1>{u['title']} &nbsp;"
        '<span style="color:#8b98a5;font-weight:400">'
        f'{shard_id}</span></h1>\n'
        f'<label class="cb">{u["your_name"]} '
        f'<input type="text" id="who" placeholder="{u["name_ph"]}" '
        'style="width:150px"></label>\n'
        '<span id="pos" style="font-size:12px;color:#8b98a5"></span>\n'
        '<div class="bar"><i id="pb"></i></div>\n'
        '<span id="cnt" style="font-size:12px;color:#8b98a5"></span>\n'
        f'<button onclick="csv()">{u["download"]}</button>\n'
        "<button onclick=\"var e=document.getElementById('cb');"
        "e.style.display=e.style.display==='none'?'block':'none'\">"
        f'{u["codebook"]}</button>\n'
        "</header>\n"
        '<div style="max-width:1180px;margin:0 auto;padding:14px 14px 0">\n'
        '<div id="banner"></div>\n'
        f'<div id="cb" style="display:none">{codebook_html}</div>\n'
        "</div>\n"
        '<div id="wrap"></div>\n'
        "<script>\n"
        f"const SHARD_ID={json.dumps(shard_id)};\n"
        f"const LABELLER={json.dumps(labeller)};\n"
        f"{_js_head(lang)}"
        f"{ITEMS_OPEN}{payload}{ITEMS_CLOSE}"
        "</script>\n"
        f"<script>{_JS}</script>\n"
        "</body></html>\n"
    )


_CB_WRAP = ("""<div style="background:#121821;border:1px solid #232c36;border-radius:8px;"""
            """padding:14px;margin-bottom:12px;font-size:13px;line-height:1.55">""")

# The codebook the labeller actually reads. Deliberately short: it is opened mid-task, on a
# parcel that is already confusing, and every sentence that is not a decision rule competes
# with the ones that are. The long-form reasoning (why UNSURE exists, why confidence was
# removed, why WOODY_NON_CROP must never fold into PERENNIAL) belongs in
# docs/s2_labelling/plan.md §3, not here.
#
# WARNING: docs/s2_labelling/codebook.md must carry the same rules. What a labeller reads and what
# is on the record cannot be allowed to drift apart.
CODEBOOK_HTML = {}

CODEBOOK_HTML["en"] = _CB_WRAP + """
<b style="font-size:14px">Codebook &mdash; pick one of six for every parcel</b>
<ul style="margin:8px 0 4px;padding-left:20px">
<li><b>1 PERENNIAL</b> &mdash; woody or multi-year crop (&gt;3 years): mango, lime, avocado,
 olive, coffee, cacao, banana, oil palm. <i>Looks like:</i> regular crowns or rows, canopy
 texture; green in both seasons of the trace. <b>Sugarcane counts as ANNUAL.</b></li>
<li><b>2 ANNUAL</b> &mdash; sown and harvested in one cycle: rice, maize, cotton, potato,
 beans, wheat. <i>Looks like:</i> uniform texture, no crowns, sharp edges; one or two NDVI
 peaks returning to bare.</li>
<li><b>3 OTHER</b> &mdash; <b>farmable land not currently cropped</b>: pasture, fallow,
 ploughed or prepared soil, weeds on sowable ground.</li>
<li><b>4 WOODY_NON_CROP</b> &mdash; trees that are not a crop: windbreaks, riparian strips,
 abandoned or invaded parcels. <i>Looks like:</i> tree cover with no rows or planting grid.
 <b>Never merge this into PERENNIAL.</b></li>
<li><b>5 UNSURE</b> &mdash; you genuinely cannot tell: cloud, deep shadow, half one thing and
 half another, imagery too coarse. <b>Use it freely</b> &mdash; an honest UNSURE is worth
 more than a guess.</li>
<li><b>6 NON_AGRICULTURE</b> &mdash; land out of agricultural use: houses and settlement,
 greenhouses and sheds, roads and tracks, canals and reservoirs, open water, riverbed sand
 and gravel, quarries, bare rock.</li>
</ul>
<b style="color:#ffdd33">The rule that separates 3 from 6</b>
<div style="margin:4px 0 8px">Ask: <b>could this ground be sown next season exactly as it
stands?</b> &nbsp;<b>Yes</b> &rarr; <b>OTHER</b>. &nbsp;<b>No</b> &mdash; something would have
to be demolished, dug up or drained first, or it is permanently water, rock or pavement
&rarr; <b>NON_AGRICULTURE</b>.<br>
Dry fallow and ploughed soil are OTHER; riverbed sand and quarry floor are NON_AGRICULTURE.
Scrub on flat farmable ground is OTHER; scrub on bare rocky slope that was never a field is
NON_AGRICULTURE. Grazed pasture is OTHER however rough it looks.</div>
<b>Order of decision</b>
<div style="margin:4px 0">Work down and stop at the first line covering &gt;50 % of the
parcel.</div>
<ol style="margin:6px 0 4px;padding-left:22px">
<li>woody / multi-year <b>crop</b> &rarr; <b>1 PERENNIAL</b></li>
<li>sown-and-harvested <b>crop</b> &rarr; <b>2 ANNUAL</b></li>
<li>trees or shrubs that are <b>not</b> a crop &rarr; <b>4 WOODY_NON_CROP</b>
 <i>(before 6)</i></li>
<li>surface that could not be sown as it stands &rarr; <b>6 NON_AGRICULTURE</b></li>
<li>farmable ground, not currently cropped &rarr; <b>3 OTHER</b></li>
<li>you cannot tell &rarr; <b>5 UNSURE</b></li>
</ol>
<b>Difficult cases</b>
<ul style="margin:6px 0 4px;padding-left:20px">
<li>mixed parcel &rarr; the class covering &gt;50 %; if genuinely even, <b>UNSURE</b>. A
 house or shed inside a cropped field does not make the parcel NON_AGRICULTURE;</li>
<li><b>young plantings</b> &rarr; PERENNIAL if a planting grid is legible <i>or</i> the trace
 keeps low green through both dry seasons; otherwise <b>UNSURE</b>. <b>Never ANNUAL by
 default</b>;</li>
<li>boundary disagrees with the outline &rarr; label what is <i>inside the outline</i> and
 tick <b>boundary mismatch</b> (<kbd>b</kbd>).</li>
</ul>
<div style="margin-top:8px;color:#8b98a5"><b>The two images.</b> Left: the whole parcel and
what it sits in (river, town edge, forest, block of fields). Right: same centre, 200 m
across, for texture &mdash; regular crowns on a grid means a planted orchard, irregular blobs
mean woody non-crop. Each panel has its own scale bar. Most imagery is 1.2 m, so the zoom
enlarges rather than resolves: if you still cannot tell, that is <b>UNSURE</b>.<br><br>
<b>The trace.</b> The line is the parcel median greenness per date. The shaded band is the
middle half of its own pixels (p25&ndash;p75), <i>not</i> an error bar: narrow means the
parcel does the same thing throughout; persistently wide means it is internally varied
&mdash; crowns against bare inter-row, or genuinely half and half, which is a prompt to
apply the &gt;50 % rule or press <kbd>5</kbd>.<br><br>
&#9888; <b>Do not look anything up.</b> The parcel has an old crop declaration and it is
deliberately not shown to you.</div>
<div style="margin-top:10px;padding-top:8px;border-top:1px solid #232c36">
<b style="color:#4ade80">When you finish</b> &mdash; once every parcel in this file is
labelled, press <b>Download CSV</b> and send the file back. Your progress is saved only in
this browser, so the CSV is the only copy that reaches anyone.</div>
</div>
"""

CODEBOOK_HTML["es"] = _CB_WRAP + """
<b style="font-size:14px">Manual &mdash; elija una de seis opciones para cada parcela</b>
<ul style="margin:8px 0 4px;padding-left:20px">
<li><b>1 PERENNE</b> &mdash; cultivo leñoso o plurianual (&gt;3 años): mango,
 limón, palto, olivo, café, cacao, plátano, palma aceitera. <i>Se ve:</i>
 copas regulares o hileras, textura de dosel; verde en las dos temporadas de la curva.
 <b>La caña de azúcar va como ANUAL.</b></li>
<li><b>2 ANUAL</b> &mdash; se siembra y cosecha en un ciclo: arroz, maíz,
 algodón, papa, frijol, trigo. <i>Se ve:</i> textura uniforme, sin copas, bordes
 nítidos; uno o dos picos de NDVI que vuelven a suelo desnudo.</li>
<li><b>3 OTRO</b> &mdash; <b>tierra cultivable sin cultivo actual</b>: pasto, barbecho,
 suelo arado o preparado, maleza sobre terreno sembrable.</li>
<li><b>4 LEÑOSO NO CULTIVO</b> &mdash; árboles que no son cultivo: cortinas
 rompevientos, franjas ribereñas, parcelas abandonadas o invadidas. <i>Se ve:</i>
 cobertura arbórea sin hileras ni marco de plantación.
 <b>Nunca lo junte con PERENNE.</b></li>
<li><b>5 NO SEGURO</b> &mdash; de verdad no se puede saber: nube, sombra, mitad y mitad,
 imagen demasiado gruesa. <b>Úselo sin problema</b>: una duda honesta vale más
 que una adivinanza.</li>
<li><b>6 NO AGRÍCOLA</b> &mdash; terreno fuera de uso agrícola: casas y poblado,
 invernaderos y galpones, carreteras y trochas, canales y reservorios, agua, cauce de
 río con arena o grava, canteras, roca desnuda.</li>
</ul>
<b style="color:#ffdd33">La regla que separa 3 de 6</b>
<div style="margin:4px 0 8px">Pregúntese: <b>¿se podría sembrar este suelo
la próxima campaña tal como está?</b> &nbsp;<b>Sí</b> &rarr;
<b>OTRO</b>. &nbsp;<b>No</b> &mdash; habría que demoler, excavar o drenar primero, o es
agua, roca o pavimento permanente &rarr; <b>NO AGRÍCOLA</b>.<br>
Barbecho seco y suelo arado son OTRO; arena de cauce y piso de cantera son NO AGRÍCOLA.
Maleza en terreno plano cultivable es OTRO; maleza en ladera rocosa que nunca fue chacra es
NO AGRÍCOLA. El pasto pastoreado es OTRO por más rústico que se vea.</div>
<b>Orden de decisión</b>
<div style="margin:4px 0">Baje por la lista y pare en la primera línea que cubra
&gt;50 % de la parcela.</div>
<ol style="margin:6px 0 4px;padding-left:22px">
<li><b>cultivo</b> leñoso o plurianual &rarr; <b>1 PERENNE</b></li>
<li><b>cultivo</b> de siembra y cosecha &rarr; <b>2 ANUAL</b></li>
<li>árboles o arbustos que <b>no</b> son cultivo &rarr; <b>4 LEÑOSO NO
 CULTIVO</b> <i>(antes que 6)</i></li>
<li>superficie que no se podría sembrar así como está &rarr; <b>6 NO
 AGRÍCOLA</b></li>
<li>terreno cultivable sin cultivo actual &rarr; <b>3 OTRO</b></li>
<li>no se puede saber &rarr; <b>5 NO SEGURO</b></li>
</ol>
<b>Casos difíciles</b>
<ul style="margin:6px 0 4px;padding-left:20px">
<li>parcela mixta &rarr; la clase que cubre &gt;50 %; si está pareja, <b>NO SEGURO</b>.
 Una casa o galpón dentro de una chacra no la hace NO AGRÍCOLA;</li>
<li><b>plantaciones jóvenes</b> &rarr; PERENNE si se ve el marco de plantación
 <i>o</i> la curva mantiene verde bajo en las dos secas; si no, <b>NO SEGURO</b>. <b>Nunca
 ANUAL por defecto</b>;</li>
<li>el borde visible no coincide con el contorno &rarr; etiquete lo que está
 <i>dentro del contorno</i> y marque <b>borde no coincide</b> (<kbd>b</kbd>).</li>
</ul>
<div style="margin-top:8px;color:#8b98a5"><b>Las dos imágenes.</b> Izquierda: la
parcela completa y su entorno (río, borde de pueblo, bosque, bloque de chacras).
Derecha: el mismo centro a 200 m de ancho, para ver textura &mdash; copas regulares en marco
= huerto plantado; manchas irregulares = leñoso no cultivo. Cada panel tiene su propia
barra de escala. La mayoría de imágenes son de 1,2 m, así que el
acercamiento agranda pero no revela más detalle: si aún no se distingue, es
<b>NO SEGURO</b>.<br><br>
<b>La curva.</b> La línea es la mediana de verdor de la parcela en cada fecha. La banda
sombreada es la mitad central de sus propios píxeles (p25&ndash;p75), <i>no</i> un
margen de error: angosta = la parcela hace lo mismo en todas partes; ancha y persistente =
parcela dispareja (copas contra suelo entre hileras, o mitad y mitad), señal para
aplicar la regla del &gt;50 % o pulsar <kbd>5</kbd>.<br><br>
&#9888; <b>No consulte nada externo.</b> La parcela tiene una declaración de cultivo
antigua y a propósito no se le muestra.</div>
<div style="margin-top:10px;padding-top:8px;border-top:1px solid #232c36">
<b style="color:#4ade80">Al terminar</b> &mdash; cuando haya etiquetado <b>todas</b> las
parcelas de este archivo, pulse <b>Descargar CSV</b> y envíe el archivo. Su avance se
guarda solo en este navegador, así que el CSV es la única copia que llega.</div>
</div>
"""


# ------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------
def build(sample: pd.DataFrame, chip_dir: Path, px: pd.DataFrame, out_dir: Path,
          shard_size: int = SHARD_SIZE, seed: int = 20260812,
          lang: str = "en") -> pd.DataFrame:
    """Emit every shard, plus the item_id -> COD_PREDIO key the ingest joins on.

    Shard assignment follows the campaign's structure (§2.1): the main parcels are split
    into shards divided between the two labellers, and the 100 double-labelled overlap
    parcels go into **one further shard that both labellers receive**. That keeps the kappa
    set from depending on how the two happened to divide the work, and makes it a single
    file that can be handed to a third person if adjudication is needed.

    Item order inside every shard is randomised with a per-shard seed, so the department
    and class structure of the draw is not legible as an ordering.
    """
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = build_traces(px)

    plan = _shard_plan(sample, shard_size)
    rows: list[dict] = []
    dropped: list[dict] = []
    for shard_id, (rows_df, labellers) in plan.items():
        base = build_items(rows_df, chip_dir, traces, dropped)
        for lab in labellers:
            # Seeded per (shard, labeller), not per shard. The overlap set goes to both
            # labellers, and giving them the same order would let fatigue and drift line
            # up between them — inflating kappa for a reason that has nothing to do with
            # the codebook.
            rng = np.random.default_rng(
                _stable_seed(f"{shard_id}|{lab}") ^ seed)
            items = [base[k] for k in rng.permutation(len(base))]
            html = render_html(items, shard_id, lab, lang=lang)
            f = out_dir / f"{shard_id}_{lab}.html"
            f.write_text(html, encoding="utf-8")
            mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {len(items)} items, {mb:.1f} MB"
                  + ("  ⚠️ OVER 16 MB" if mb > 16 else ""))
        for it in base:
            rows.append({"item_id": it["item_id"], "shard": shard_id,
                         "suggested_labellers": ";".join(labellers)})

    key = (pd.DataFrame(rows)
           .merge(sample[["item_id", "COD_PREDIO"]], on="item_id", how="left"))
    key.to_csv(out_dir / "item_key.csv", index=False)

    n_items = key["item_id"].nunique()
    print(f"\nwrote {len(plan)} shards + item_key.csv to {out_dir}")
    print(f"  {n_items:,} of {len(sample):,} drawn parcels are labellable")
    if dropped:
        dd = pd.DataFrame(dropped).drop_duplicates(["item_id", "reason"])
        dd.to_csv(out_dir / "excluded_from_shards.csv", index=False)
        print("  attrition (written to excluded_from_shards.csv):")
        print(dd.groupby("reason").size().to_string())
    return key


def _shard_plan(sample: pd.DataFrame, shard_size: int) -> dict:
    """``shard_id -> (rows, [labellers])`` — the pilot, the four main shards, the overlap."""
    plan: dict[str, tuple] = {}
    pilot = sample[sample.batch == "pilot"]
    if len(pilot):
        plan["pilot"] = (pilot, ["A", "B"])
    main = sample[sample.batch == "main"].reset_index(drop=True)
    overlap = main[main.overlap]
    solo = main[~main.overlap].reset_index(drop=True)
    n_shards = max(1, int(np.ceil(len(solo) / shard_size)))
    # Greedy least-loaded, not "first half to A": with an odd shard count a positional
    # split hands one labeller 600 parcels and the other 292.
    load = {"A": 0, "B": 0}
    for k in range(n_shards):
        rows = solo.iloc[k * shard_size:(k + 1) * shard_size]
        lab = min(load, key=lambda x: load[x])
        load[lab] += len(rows)
        plan[f"shard{k + 1:02d}"] = (rows, [lab])
    if len(overlap):
        plan["overlap"] = (overlap, ["A", "B"])
    return plan
