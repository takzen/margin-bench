<p align="center"><img src="assets/banner.svg" alt="margin-bench: can a language model design to spec?" width="100%"></p>

# margin-bench

Can a language model design to spec?

margin-bench gives models engineering design tasks in three domains: digital signal
processing, electronics and control systems. The model gets a specification and answers
with a design: filter taps, fixed-point coefficients, component values, controller
gains. The answer is graded by measurement or simulation (scipy, python-control,
ngspice), never by another model. A filter either meets its mask or it doesn't, and the
benchmark shows by how much.

[![tests](https://github.com/takzen/margin-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/takzen/margin-bench/actions/workflows/tests.yml)

**Status:** all nine task families of the first version are implemented and tested. No
model has been run yet, so there are no results. One family has been reviewed and
accepted by the author; the other eight are awaiting review.

## How a task is graded

Take `dsp.fir_lowpass_mask`. The prompt gives a sampling rate, passband and stopband
edges, a passband ripple limit, a stopband attenuation requirement and a maximum number
of taps. The model answers with `{"taps": [...]}`. The grader computes the magnitude
response on 4096 frequencies per band, edges included, and compares it with the mask.

* **Pass/fail:** every requirement met.
* **Score from 0 to 1:** each requirement gives a ratio that reaches 1 exactly when it
  is met (achieved attenuation / required attenuation; ripple limit / achieved ripple).
  The score is the smallest ratio, so it is 1.000 if and only if the answer passes, and
  otherwise shows how far the weakest requirement fell short.
* **Format errors** score 0 and are counted separately, so "cannot design it" and
  "cannot follow the answer format" stay apart. Breaking a hard constraint (too many
  taps, a resistor that is not in the E24 series) also scores 0 but is not a format
  error.

Every attempt keeps plot data: the response drawn against the spec, with violations in
red. The site shows it for every attempt.

## Task families

| Domain | Family | What the model designs | Graded by | Review |
|---|---|---|---|---|
| DSP | `dsp.fir_lowpass_mask` | FIR lowpass taps for a magnitude mask and a tap limit | `scipy.signal.freqz` on a dense grid | accepted |
| DSP | `dsp.q15_biquad` | int16 biquad coefficients and shift (Direct Form I, int32 accumulator) for an RBJ lowpass | magnitude and DC gain of the integer filter; exact integer simulation with the worst-case input for overflow | awaiting review |
| DSP | `dsp.ema_shift` | `acc += ((x << F) - acc) >> k`: shift, fractional bits, output rounding | cutoff of the linearized filter; settled error from both rails (the dead band); int32 headroom | awaiting review |
| Electronics | `electronics.sallen_key_e24` | unity-gain Sallen-Key from E24 resistors and E12 capacitors for f0 and Q | ngspice AC sweep: f0 at -90°, Q = gain there | awaiting review |
| Electronics | `electronics.opamp_gain_bw` | up to three non-inverting stages (E96) for a gain and a bandwidth with a finite-GBW op-amp | ngspice AC sweep: gain at 1 Hz, -3 dB bandwidth | awaiting review |
| Electronics | `electronics.divider_tolerance` | a loaded divider (E24 or E96) that holds its output window at every tolerance corner under a current limit | ngspice operating point at 16 corners | awaiting review |
| Control | `control.pid_step_spec` | PID gains for overshoot, settling time, peak control signal and steady-state error | python-control step responses of the closed loop | awaiting review |
| Control | `control.loop_margins` | lead/lag compensator for phase margin, gain margin and crossover frequency | python-control `stability_margins`, closed-loop poles | awaiting review |
| Control | `control.discrete_pid` | discrete PID coefficients at a sample time | ZOH-sampled closed loop against the continuous loop | awaiting review |

Each family has a reference answer that passes and a known-bad answer that fails. The
known-bad answers are realistic mistakes: a truncated sinc without a window, a Q15
biquad whose a1 does not fit and saturates, an EMA without fractional bits, a
proportional-only controller, a pure gain where phase lead is needed, an integral term
that forgot the sample time, equal-component Sallen-Key (Q stuck at 0.5), one
op-amp stage with the whole gain, a divider that ignores its load. The test suite
checks the pair on 30 to 100 seeds per family.

How hard each family is comes from a written rule about design methods, never from a
model's score. The rule is at the top of each family's module; for example, the tap
limit in `dsp.fir_lowpass_mask` is Kaiser's length estimate times 1.1, so a
Kaiser-window design with sensible margin always fits.

## Rules the benchmark keeps

1. **No LLM as judge.** Grading is deterministic measurement or simulation.
2. **Tasks are generated from a seed.** The `dev` set (seeds 0 and up) is public. The
   `test` set is hidden and drawn again for every benchmark version (see below).
   Generators use only `random.Random.random()`, whose sequence Python guarantees across
   versions; a test pins the first `dev` instance, and the export refuses to publish
   results whose prompts no longer match the generator.
3. **Continuous score and pass/fail**, both reported.
4. **k attempts per task.** Results are means with 95% bootstrap intervals. The
   bootstrap resamples task instances rather than attempts, because attempts on the
   same task are correlated. The overall score weights families equally. Every attempt
   records its tokens (reasoning included) and cost.
5. **One prompt for every model.** Sampling settings (reasoning effort, max tokens,
   temperature, pinned provider) are stored with each attempt.
6. **Reference passes, negative fails**, checked by the test suite for every family.
7. **Model code runs only in a sandbox** (for later code-answer tasks; none yet).

## Conflict of interest

This project is built with help from Claude, and Claude models are among those
evaluated. To keep that from tilting the results:

* task difficulty is set by rules about design methods, never by any model's score;
* difficulty is checked on models from several providers, and tasks are never tuned to
  one model's results, Claude's least of all;
* every task family is approved by the author, who works in these fields;
* raw transcripts are published with the results.

## Running it

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and the ngspice shared library
for the electronics families. On Windows the copy bundled with KiCad is found
automatically; elsewhere install it (`apt install libngspice0` on Debian/Ubuntu) or set
`NGSPICE_PATH` to `ngspice.dll` / `libngspice.so`. Without ngspice the electronics tests
are skipped.

```sh
uv sync
uv run pytest                                             # references pass, negatives fail
uv run margin-bench families
uv run margin-bench prompt dsp.q15_biquad --seed 0        # the exact prompt a model sees
```

### Running models

Models are called through [OpenRouter](https://openrouter.ai). Put the key in `.env`
(gitignored):

```
OPENROUTER_API_KEY=sk-or-...
```

A run is described by a TOML file in `runs/`. Without `--execute`, `run` only prints a
cost estimate from OpenRouter's public price list and calls nothing:

```sh
uv run margin-bench run runs/smoke-all.toml               # dry run: every family once, 4 models
uv run margin-bench run runs/smoke-all.toml --execute     # paid
uv run margin-bench report results/runs/smoke-all.jsonl
```

| Config | Calls | Estimated cost | Worst case | Budget cap |
|---|---|---|---|---|
| `runs/smoke-fir.toml` | 40 | $3.72 | $9.90 | $10 |
| `runs/smoke-all.toml` | 36 | $3.35 | $8.91 | $10 |
| `runs/dev-v0.toml` | 432 | $40.20 | $106.92 | $60 |

Estimates assume 12,000 output tokens per call; the worst case assumes every call
hits `max_tokens`. Prices are OpenRouter list prices on 2026-10-15.

Each attempt is appended to a JSONL file as it finishes, so an interrupted run resumes
where it stopped. Failed API calls are recorded and retried on the next run; they never
count as model failures. No new call starts once `max_cost_usd` is spent. Every request
sets OpenRouter's `require_parameters`, so a provider that would silently ignore a
setting is never used.

### The hidden test set

```sh
uv run margin-bench new-test-set test-v0 --per-family 10   # secret seeds -> private/test-v0.json
```

A run config then uses `seeds_file = "private/test-v0.json"` instead of `seeds`, and must
write its output under `private/`. Seeds come from the operating system's secure random
source and sit above the dev range. `private/` is gitignored: keep a backup elsewhere.
`export-site` refuses results from `private/` unless `--publish-hidden` is given, which
should happen only when that benchmark version is final.

### The site

```sh
uv run margin-bench export-site results/runs/dev-v0.jsonl  # writes site/data
uv run python -m http.server -d site                      # http://localhost:8000
```

`site/` is plain HTML, JavaScript and SVG with no build step; it can be served by any
static host (for Cloudflare Pages: no build command, output directory `site`). Until a
real run exists, `site/data` holds demo data from `margin-bench export-demo`: each
family's reference and negative answers graded like model answers, clearly labeled.

## Layout

```
src/margin_bench/
  core.py        Task, Score, the TaskFamily protocol
  tasks/         task families (generator, grader, reference, negative), by domain
  grading/       answer parsing and validation, continuous scoring, plot data format
  sims/          ngspice through its shared library
  harness/       OpenRouter client, run configs, attempt store, cost estimate
  report/        aggregation with bootstrap intervals, export for the site
tests/
runs/            run configs
results/         raw attempts (JSONL)
site/            static site and its data
private/         hidden test set seeds and results (gitignored)
```

## License

MIT, see [LICENSE](LICENSE). It covers the code, the task generators, the run configs and the
published results.
