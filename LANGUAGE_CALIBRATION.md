# Language Calibration Guide

The calibration set in `scripts/generate_calib.py` should match the language(s)
your application will use in production.

Activation ranges differ significantly between scripts, so a model calibrated
only on English may produce degraded output quality for other languages.

## Why it matters

During W8A8 quantisation, `hybrid_quantize()` runs forward passes through every
layer using your calibration samples and records the min/max activation values
per channel. Those ranges become the quantisation scales baked into the `.rkllm`
file. If a language's token embeddings were never seen during calibration, the
scales will be wrong for that language and quality will suffer.

## How to adapt for your language

Replace or extend `CALIBRATION_PROMPTS` in `scripts/generate_calib.py` with
samples from your target language. Aim for 3-5 samples per language, 20-64
total. After editing, re-run the two affected steps:

```bash
docker compose run --rm convert <model_id> <res>
# Steps 4 (calib) and 5 (rkllm) will re-run; earlier steps are cached.
```

## Language-specific notes

**Arabic** - use right-to-left text with diacritics and connected letter forms.
Include both Modern Standard Arabic and colloquial if your app uses both.

**Japanese** - mix kanji, hiragana, katakana and full-width punctuation.
A set that only contains one script will miss the activation ranges of the others.

**Korean** - use Hangul with correct spacing rules (spaces between words, not
between syllables). Single-syllable samples are not representative.

**Russian / Cyrillic** - include both printed vocabulary and technical terms.
Cyrillic embeddings are well-represented in Qwen but still benefit from
explicit calibration samples.

**German** - include compound nouns (e.g. Bundesverfassungsgericht) and umlauts
(ä, ö, ü, ß). Compounds tokenise very differently from space-separated words.

**Hebrew** - right-to-left, include niqqud if your use case involves formal or
educational text. Casual Hebrew without niqqud tokenises differently.

**Thai** - no spaces between words. Include multi-word spans, not isolated words.
Single-word samples do not reflect real sentence activation patterns.

## Recommended workflow for a custom language

1. Collect 10-20 real prompt/response pairs from your actual application
2. Add them to `CALIBRATION_PROMPTS` in the same `{"prompt": ..., "completion": ...}` format
3. Delete the cached `data_quant.json` in the model output folder so it is regenerated:
   ```bash
   rm output/<model-name>/data_quant.json
   ```
4. Re-run the conversion - steps 1-3 will be skipped (cached), steps 4-5 will re-run
5. Compare output quality on a held-out test set before and after