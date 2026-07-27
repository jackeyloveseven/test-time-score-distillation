# Human study kit (blind style-transfer rating)

Buildable-without-annotators infra for the AAAI paper's human-eval (Limitations lists it as future work).

- `rate.html` — open locally in a browser (from this `human_study/` dir so `../results/*.png` resolve). Presents 14 (reference, generated) pairs in random order; rater scores each on Style-match / Content-clarity / Overall (1-5 Likert), blind to CSD. Exports `style_ratings.json`.
- `pairs.json` — the curated pair manifest (style id + paths + CSD for the owner; CSD is NOT shown to raters).

To run a study: send `rate.html` + the referenced PNGs (or the whole repo) to N raters, collect their `style_ratings.json`, average per-axis. Design = absolute Likert (no baseline needed), which validates the 'CSD under-rates; humans confirm quality' claim independently of the uncalibrated CSD metric.
