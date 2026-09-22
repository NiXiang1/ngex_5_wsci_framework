# WSCI Framework - Results and Discussion

## Reproduction

The final implementation was verified with the locally installed `qwen3:8b` model:

```powershell
ollama pull qwen3:8b
python -m pip install -r requirements.txt
python bad_wsci.py --model qwen3:8b
python manual_wsci.py --model qwen3:8b
python smarter_wsci.py --model qwen3:8b
python -m unittest discover -s tests -v
```

The model can be changed with `--model` or `OLLAMA_MODEL`; no source edit is
required.

## Observed context sizes

| Program | Context approach | Characters (`qwen3:8b`) |
| --- | --- | ---: |
| `bad_wsci.py` | All seven knowledge files | 4,222 |
| `manual_wsci.py` | Three manually selected files | 2,176 |
| `smarter_wsci.py` | Three automatically selected files | 2,176 |
| `smarter_wsci.py` | Qwen-compressed structured context | 812 |

Manual selection reduced the supplied context by about 48.5% without changing the
main diagnosis. Compression reduced the selected context by a further 62.7%, or
about 80.8% relative to the all-files baseline.

## Result quality

All three versions identified cached Windows Wi-Fi credentials as the likely cause.
The common recommendation was to forget `eduroam` and reconnect with the updated
university password. The smart pipeline selected:

- `wifi_setup.txt`
- `password_changes.txt`
- `service_status.txt`

It produced schema-constrained JSON and wrote the result to `state.json`. On later
runs, only the previous answer from the `wifi_setup` partition was reused; unrelated
diagnostic partitions and report counters were not sent to the model.

## Model-size comparison

An earlier run in this repository used `qwen3:0.6b` and produced a 526-character
compressed context. The final `qwen3:8b` run produced 812 characters. Both models
reached the same core diagnosis and action, but their wording and degree of detail
differed. This is expected: WSCI controls what information is supplied, while model
size still affects how that information is summarized and expressed.

The implementation therefore treats the model name as runtime configuration and
uses temperature 0 plus JSON schemas for repeatability. Schema validation guarantees
the output structure, not factual correctness; grounding still depends on careful
prompts and the supplied knowledge files.

## Verification outcome

Seven offline tests passed. They cover automatic selection, result limits, invalid
JSON shapes, rejection of extra fields, state isolation, report counters, and an
end-to-end mocked pipeline. A live `qwen3:8b` run also completed successfully and
updated the submitted `state.json` artifact.
