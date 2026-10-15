<p align="center"><img src="assets/banner.svg" alt="margin-bench: can a language model design to spec?" width="100%"></p>

# margin-bench

Can a language model design to spec?

margin-bench gives models engineering design tasks in three domains: digital signal
processing, electronics and control systems. The model gets a specification and answers
with a design: filter taps, component values, controller gains. The answer is graded by
measurement or simulation (scipy, python-control, ngspice), never by another model. A
filter either meets its mask or it doesn't, and the benchmark shows by how much.

**Status:** early development. No model has been run yet, so there are no results.
Everything below describes the method and the code as they stand.

## How a task is graded

Take the first task family, `dsp.fir_lowpass_mask`. The prompt gives a sampling rate,
passband and stopband edges, a passband ripple limit, a stopband attenuation
requirement and a maximum number of taps. The model answers with `{"taps": [...]}`. The
grader computes the magnitude response on 4096 frequencies per band, edges included,
and compares it with the mask.

* **Pass/fail:** every point inside the mask, and no more taps than allowed.
* **Score from 0 to 1:** each requirement gives a ratio that reaches 1 exactly when it
  is met (achieved attenuation / required attenuation; ripple limit / achieved ripple).
  The score is the smallest ratio, so it is 1.000 if and only if the answer passes, and
  otherwise shows how far the weakest requirement fell short.
* **Format errors** score 0 and are counted separately, so "cannot design it" and
  "cannot follow the answer format" stay apart.

Every attempt keeps its plot data: the response drawn against the mask, with
violations marked. The public site will show this for every attempt.

## Rules the benchmark keeps

1. **No LLM as judge.** Grading is deterministic measurement or simulation.
2. **Tasks are generated from a seed.** The `dev` set is public. The `test` set is
   hidden and drawn again for every benchmark version; its seeds never enter this repo.
   Generators use only `random.Random.random()`, whose sequence Python guarantees across
   versions, and a test pins the first `dev` instance so it cannot change silently.
3. **Continuous score and pass/fail**, both reported.
4. **k attempts per task.** Results are means with 95% bootstrap intervals. The
   bootstrap resamples task instances rather than single attempts, because attempts on
   the same task are correlated. Every attempt records its tokens (reasoning included)
   and its cost.
5. **One prompt for every model.** Sampling settings (reasoning effort, max tokens,
   temperature, pinned provider) are stored with each attempt.
6. **Every task family has a reference answer that passes and a known-bad answer that
   fails**, both checked by the test suite on 100 seeds. A family without them does not
   enter the benchmark.
7. **Model code runs only in a sandbox** (for the later code-answer tasks).

## Conflict of interest

This project is built with help from Claude, and Claude models are among those
evaluated. To keep that from tilting the results:

* task difficulty is set by rules about design methods, never by any model's score
  (for example: the tap limit in `dsp.fir_lowpass_mask` is Kaiser's length estimate
  times 1.1, so a Kaiser-window design with sensible margin always fits);
* difficulty is checked on models from several providers, and tasks are never tuned to
  one model's results, Claude's least of all;
* every task family is approved by the author, who works in these fields;
* raw transcripts are published with the results.

## Task families

| Domain | Family | Status |
|---|---|---|
| DSP | `dsp.fir_lowpass_mask`: FIR lowpass taps that fit a magnitude mask | accepted |
| DSP | `dsp.q15_biquad`: biquad coefficients in Q15 without overflow | planned |
| DSP | embedded DSP from practice (fixed-point EMA, lock-in) | planned |
| Electronics | `electronics.sallen_key_e24`: Sallen-Key fc and Q from E24 values | planned |
| Electronics | `electronics.opamp_gain_bw`: gain and bandwidth at a given GBW | planned |
| Electronics | `electronics.divider_tolerance`: worst-case divider with tolerances | planned |
| Control | `control.pid_step_spec`: PID for overshoot, settling time, steady-state error | planned |
| Control | `control.loop_margins`: compensator for phase and gain margin | planned |
| Control | `control.discrete_pid`: discretized controller at a given Ts | planned |

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                                             # references pass, negatives fail
uv run margin-bench families
uv run margin-bench prompt dsp.fir_lowpass_mask --seed 0  # the exact prompt a model sees
```

Models are called through [OpenRouter](https://openrouter.ai). A run is described by a
TOML file in `runs/`. Without `--execute`, `run` only prints a cost estimate from
OpenRouter's public price list and calls nothing:

```sh
uv run margin-bench run runs/smoke-fir.toml              # dry run with cost estimate
uv run margin-bench run runs/smoke-fir.toml --execute    # paid; needs OPENROUTER_API_KEY in .env
uv run margin-bench report results/runs/smoke-fir.jsonl
```

Each attempt is appended to a JSONL file as it finishes, so an interrupted run resumes
where it stopped. A run config can set `max_cost_usd`; no new calls start once it is
spent. Every request sets OpenRouter's `require_parameters`, so a provider that would
silently ignore a setting is never used.

## Layout

```
src/margin_bench/
  core.py        Task, Score, the TaskFamily protocol
  tasks/         task families: generator, grader, reference, negative
  grading/       answer parsing and continuous scoring
  harness/       OpenRouter client, run configs, attempt store, cost estimate
  report/        aggregation with bootstrap intervals
tests/
runs/            run configs
results/         raw attempts (JSONL) and aggregates
site/            static site: leaderboard and attempt viewer
```

## License

Not chosen yet. Until a license file is added, the default copyright rules apply.
