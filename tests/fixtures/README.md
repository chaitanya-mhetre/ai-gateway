# Provider fixtures

These are **hand-written from each provider's documented wire format** (checked against current
docs in Sep 2026). They are *not* captures of real traffic. Where a format detail was uncertain,
the adapter is written defensively (e.g. Gemini tool-call ids are optional).

To refresh with real recordings: run `pytest -m live` with real keys, and save the raw bodies here.
