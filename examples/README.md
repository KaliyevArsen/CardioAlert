# Fictional demonstrations

Both JSON files are manually constructed fictional inputs. They are not patient
records, pilot observations or data used to measure model accuracy.

After running the Cleveland training protocol:

```bash
cardioalert predict --input examples/fictional_complete.json
cardioalert predict --input examples/fictional_missing.json
```

The second example demonstrates explicit missing-value handling. Numeric input
codes are documented in [data and methods](../docs/DATA_AND_METHODS.md).
An input score should never be read as a clinical diagnosis or a future risk
percentage.

The generated `demo_outputs.json` and `bot_transcript.txt` show these same
fictional examples processed by the reconstructed model/state machine. They
demonstrate software behavior, not real-world performance or live Telegram use.
