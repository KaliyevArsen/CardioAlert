# Project history and evidence

## Preserved material

The public repository before restoration contained a training notebook and a
Telegram bot. Both originals remain byte-for-byte unchanged in `archive/`.
The Git history remains intact. They are historical references, not the current
supported execution path.

Source commit: [`2a5f7630c2d99bbec561e51a08d98255a4691049`](https://github.com/KaliyevArsen/CardioAlert/commit/2a5f7630c2d99bbec561e51a08d98255a4691049).

| Preserved artifact | Git blob SHA |
|---|---|
| `archive/legacy_training.ipynb` | `49983fcb3d91305bd5b58e90985d8a5331c2627d` |
| `archive/legacy_bot.py` | `9407f794309b922560a3a74548c0a0ea3048d4e3` |

The notebook's saved output records a 61-example Cleveland holdout,
accuracy **0.8688524590163934** and ROC-AUC **0.902056277056277**. It also contains
an out-of-order execution error and exploratory cells. Preserving those cells
does not mean the archived notebook is the recommended quick start.

## Restoration

The current package reconstructs training, evaluation, examples and the bot
interface from the published code. It adds shared schema validation, automated
tests, reproducible reports and a separate four-site public-data experiment.
This is new restoration work, not a recovered copy of unavailable original files.

Results are labeled by their actual provenance:

| Item | What this repository supports |
|---|---|
| Original 61-example results | Saved outputs in the preserved notebook |
| Current Cleveland results | Newly executed fixed-protocol report |
| Current 300-example results | Newly executed, separately labeled multisite UCI benchmark |
| JSON examples and bot transcripts | Explicitly fictional software demonstrations |
| Historical supervised pilot | No original pilot records or protocol are included; demo examples do not verify a pilot or its size |
| Daryn award and utility-model patent | Project background reported by the author; independent award/patent documents are not model-validation evidence |

The author's [public profile](https://kz.linkedin.com/in/arsen-kaliyev) lists
third place at the Daryn national scientific-project final and Kazakhstan
utility-model patent **No. 11198**, titled *Method for Cardiovascular Disease Risk
Assessment Using a Server-Based Inferential System*. The patent concerns the
project; the benchmark data are from the credited UCI creators. A patent or an
award does not establish clinical efficacy.

If historical pilot records or another original evaluation are recovered, add
their protocol, dates, consent/governance context, de-identified aggregate results
and provenance separately. Do not relabel a new benchmark or fabricated records
as historical activity. Public releases must not expose private health data.

## Contributions

Project owner: **Arsen Kaliyev**. The author describes his original role as model
development and evaluation, with a teammate contributing to the Telegram
interface. The current restoration reorganizes and updates the published code;
it does not establish additional historical contributors, team size or duties.
