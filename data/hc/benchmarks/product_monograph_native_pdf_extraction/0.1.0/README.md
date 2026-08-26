# Product Monograph full-PDF extraction 0.1.0

This directory contains the compact dataset artifacts for the full-PDF Product
Monograph condition. It reuses the frozen 40-family bilingual cohort and the
same 390 reference facts as the evidence-window condition, but each case points
to one complete hash-locked official PDF instead of embedding selected page
text.

Tracked artifacts contain case descriptors and verification metadata, not the
source PDF bytes. Reference facts have automated source-page evidence and no
independent human sign-off, so benchmark results are reported descriptively.

See [`docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md`](../../../../../docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
for the input contract, evidence boundary, results, and rebuild commands.
