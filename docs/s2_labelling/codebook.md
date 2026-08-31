# Manual de etiquetado — campaña S2 (codebook)

> **Frozen 2026-08-13, shortened and translated 2026-08-14, before any parcel was labelled.**
> The same rules are embedded in every labelling HTML behind the *Manual* button, so what a
> labeller reads and what is on the record here cannot drift apart. The delivered set is
> **Spanish** (`labels_s2/html_es/`); the English build is kept as a reference
> (`labels_s2/html/`).
>
> Source of truth: `labelling/build_html.py::CODEBOOK_HTML`, one entry per language.
> **It is deliberately short.** It is opened mid-task, on a parcel that is already
> confusing, and every sentence that is not a decision rule competes with the ones that are.
> The reasoning behind the design lives in [`plan.md`](plan.md) §3 and §5, not here.

---

## Escriba su nombre antes de empezar

La letra del nombre del archivo (`shard01_A.html`) sólo sugiere quién debería tomarlo. Lo que
entra al CSV es el nombre que usted escriba. Su avance se guarda contra ese nombre, así que
dos personas pueden usar la misma computadora sin pisarse.

Para cada parcela verá tres cosas:

1. **contexto** (izquierda) — la parcela completa con contorno **amarillo**, las vecinas en
   **celeste**, y su entorno: río, borde de pueblo, bosque, bloque de chacras;
2. **acercamiento** (derecha) — el mismo centro, **200 m de ancho**, para ver textura;
3. **curva NDVI** — 24 meses de verdor Sentinel-2, un punto por observación despejada, con
   una línea punteada en la fecha de la imagen aérea.

Cada panel tiene su propia barra de escala.

---

## Las seis opciones

| tecla | valor | cubre |
|---|---|---|
| `1` | **PERENNE** | cultivo leñoso o plurianual (>3 años): mango, limón, palto, olivo, café, cacao, plátano, palma aceitera |
| `2` | **ANUAL** | se siembra y cosecha en un ciclo: arroz, maíz, algodón, papa, frijol, trigo |
| `3` | **OTRO** | tierra cultivable sin cultivo actual: pasto, barbecho, suelo arado o preparado, maleza sobre terreno sembrable |
| `4` | **LEÑOSO NO CULTIVO** | árboles que no son cultivo: cortinas rompevientos, franjas ribereñas, parcelas abandonadas o invadidas |
| `5` | **NO SEGURO** | de verdad no se puede saber |
| `6` | **NO AGRÍCOLA** | fuera de uso agrícola: casas, invernaderos y galpones, carreteras, canales y reservorios, agua, cauce de río, canteras, roca desnuda |

**Cómo se ven:** PERENNE — copas regulares o hileras, textura de dosel, verde en las dos
temporadas de la curva. ⚠️ **La caña de azúcar va como ANUAL** (igual que en
`config/perennial_allperu.yaml`, `cana_policy: annual`, y en MapBiomas).
ANUAL — textura uniforme, sin copas, bordes nítidos; uno o dos picos que vuelven a suelo
desnudo. LEÑOSO NO CULTIVO — cobertura arbórea **sin** hileras ni marco de plantación.
⚠️ **Nunca lo junte con PERENNE:** lo leñoso no cultivo declarado es 2.79 % del universo
nacional, más grande que el efecto que este estudio quiere medir.

**NO SEGURO no es una falla y no se le cuenta en contra — úselo sin problema.** Una duda
honesta vale más que una adivinanza: la adivinanza es indistinguible de una etiqueta real
más adelante, mientras que una parcela NO SEGURO simplemente se aparta.

## La regla que separa 3 de 6 — aplíquela literalmente

Pregúntese: **¿se podría sembrar este suelo la próxima campaña tal como está?**
**Sí** → **OTRO**. **No** — habría que demoler, excavar o drenar primero, o es agua, roca o
pavimento permanente → **NO AGRÍCOLA**.

Barbecho seco y suelo arado son OTRO; arena de cauce y piso de cantera son NO AGRÍCOLA.
Maleza en terreno plano cultivable es OTRO; maleza en ladera rocosa que nunca fue chacra es
NO AGRÍCOLA. El pasto pastoreado es OTRO por más rústico que se vea.

## Orden de decisión

Baje por la lista y pare en la primera línea que cubra **>50 %** de la parcela. El orden es
lo que evita que dos personas partan la misma parcela entre 4 y 6, o entre 3 y 6.

1. **cultivo** leñoso o plurianual → **1 PERENNE**
2. **cultivo** de siembra y cosecha → **2 ANUAL**
3. árboles o arbustos que **no** son cultivo → **4 LEÑOSO NO CULTIVO** *(antes que 6 — los
   árboles ribereños son leñoso no cultivo; el agua y la grava al lado son no agrícola)*
4. superficie que no se podría sembrar así como está → **6 NO AGRÍCOLA**
5. terreno cultivable sin cultivo actual → **3 OTRO**
6. no se puede saber → **5 NO SEGURO**

## Casos difíciles

* **parcela mixta** → la clase que cubre >50 %; si está pareja, `NO SEGURO`. Una casa o
  galpón dentro de una chacra no la hace NO AGRÍCOLA — sólo si está *mayormente* construida;
* **agroforestería con dosel cerrado** → `PERENNE`;
* **plantaciones jóvenes** → `PERENNE` si se ve el marco de plantación **o** la curva
  mantiene verde bajo en las dos secas; si no, `NO SEGURO`. **Nunca `ANUAL` por defecto** —
  un huerto joven leído como anual es un falso negativo justo sobre la transición que este
  estudio mide;
* **ilegible** (nube, sombra, cobertura parcial) → `NO SEGURO`;
* **el borde visible no coincide con el contorno** → etiquete **lo que está dentro del
  contorno** y marque **borde no coincide** (`b`).

## Leer las dos imágenes

La izquierda da **forma, vecinas y entorno**: la tierra perenne suele verse distinta de las
chacras anuales de al lado, y el paisaje alrededor es lo que hace legible el mismo cultivo en
Piura y en Huancavelica. La derecha da **textura**: copas regulares en marco = huerto
plantado; manchas irregulares = leñoso no cultivo; tono uniforme = chacra anual o suelo
desnudo.

⚠️ La mayoría de imágenes son de 1,2 m, así que el acercamiento **agranda pero no revela más
detalle**. Si a 200 m de ancho todavía no se distingue, la información no está — eso es un
`NO SEGURO`, no una mirada más larga.

## Leer la curva

Es lo más útil de la página, y es lo que hace posible distinguir ANUAL de OTRO: una sola foto
aérea no separa una chacra cosechada de un barbecho.

* **barbecho / desnudo** — plana y baja en las dos temporadas;
* **anual** — uno o dos picos claros, volviendo a desnudo entre ellos;
* **perenne madura** — verde persistente con oscilación *pequeña*;
* **huerto joven** — verde de amplitud baja que nunca llega a desnudo.

El eje y está fijo entre −0,1 y 1,0 en todas las parcelas, así que son directamente
comparables.

**La banda sombreada es la mitad central de los píxeles de la propia parcela (p25–p75), no un
margen de error.** Banda angosta = la parcela hace lo mismo en todas partes (chacra uniforme,
dosel cerrado). Banda ancha y persistente = parcela dispareja — copas contra suelo entre
hileras, o de verdad mitad y mitad, señal para aplicar la regla del >50 % o pulsar `5`.

## Controles

| tecla | acción |
|---|---|
| `1` `2` `3` `4` `5` `6` | asignar la clase **y avanzar** |
| `b` | marcar/desmarcar borde no coincide |
| `←` `→` | moverse sin etiquetar |

**`NO SEGURO` es la única forma de abstenerse.** No hay control de confianza — se quitó,
porque una adivinanza con confianza baja y una etiqueta real son indistinguibles una vez
dentro del conjunto de entrenamiento. O la llama, o pulsa `5`.

⚠️ **No consulte nada externo.** La parcela tiene una declaración de cultivo de 1997–2006 y a
propósito no se le muestra. Si estuviera anclado en ella, fabricaría concordancia entre la
declaración y lo que ve, que es exactamente el error que esta campaña existe para no cometer.

## Al terminar

Cuando haya etiquetado **todas** las parcelas del archivo, pulse **Descargar CSV** y envíe el
archivo. Su avance se guarda **sólo en este navegador**, así que el CSV es la única copia que
llega. Puede descargarlo las veces que quiera.

---

## What the labeller is not told, and why

The declared PETT class, the train/test assignment and the CV fold are **not present anywhere
in the HTML file**, not merely hidden — `tests/test_build_html.py` asserts on the raw file
text, in both languages. Department is shown, because it is obvious from the imagery anyway.

**Translation touches the display name only.** The value written to the CSV, and every value
`ingest.py` compares against, stays the canonical English constant (`PERENNIAL`, `ANNUAL`,
`OTHER`, `WOODY_NON_CROP`, `UNSURE`, `NON_AGRICULTURE`). Localising the stored value would
have made the ingest match nothing and report an empty label distribution rather than an
error — the same class of silent failure as the four data traps in `DATA.md`.
