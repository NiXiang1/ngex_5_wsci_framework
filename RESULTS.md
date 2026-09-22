# WSCI Framework - Run Results

Model used: `qwen3:0.6b`

## How to run

Start Ollama and make sure the model is available:

```bash
ollama pull qwen3:0.6b
```

Install the Python client, then run each example from the repository root:

```bash
python -m pip install -r requirements.txt
python bad_wsci.py
python manual_wsci.py
python smarter_wsci.py
```

Set `OLLAMA_MODEL` to use another local Qwen model if required.

## Observed results

| Program | Context approach | Context characters |
| --- | --- | ---: |
| `bad_wsci.py` | All knowledge files | 4224 |
| `manual_wsci.py` | Three manually selected files | 2178 |
| `smarter_wsci.py` | Three automatically selected files | 2176 |
| `smarter_wsci.py` | Qwen-compressed JSON context | 526 |

All three runs correctly identified cached old Windows credentials as the most
likely cause. The recommended action was to forget the existing `eduroam`
network and reconnect with the new university password.

The smart workflow selected:

- `wifi_setup.txt`
- `password_changes.txt`
- `service_status.txt`

It wrote a structured answer to `state.json`. State is isolated by diagnostic
category, so a later Wi-Fi request receives only the `wifi_setup` partition and
not unrelated state. A second run successfully reused that relevant partition
and updated the report counters.
