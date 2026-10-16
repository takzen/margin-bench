<p align="center"><img src="assets/banner.svg" alt="margin-bench: can a language model design to spec?" width="100%"></p>

# margin-bench

Can a language model do engineering work: run a plant, commission a control loop, design to spec?

margin-bench has two tracks. Both are graded by simulation, never by another model.

* **Playground.** The model gets a simulated plant and tools to operate it. It runs experiments, reads
  the trends, changes settings and checks the result, as an engineer would, then hands over. An
  acceptance test on the plant decides. Every step, every thought the model shares and every second of
  plant time can be replayed on the site, or watched live while the model works.
* **Design tasks.** One-shot design to spec in DSP, electronics and control: filter taps, fixed-point
  coefficients, component values, controller gains, graded with scipy, python-control and ngspice.

[![tests](https://github.com/takzen/margin-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/takzen/margin-bench/actions/workflows/tests.yml)

**Status:** the playground has two environments, `process.tank_level` and `dsp.scale_readout`; more
plants will follow (a furnace with long dead time, a DC motor, an inverted pendulum, a circuit with a
fault). All nine design task families are implemented. No model has been run yet, so there are no results.

## Playground

### `process.tank_level`: commission a level loop

A vertical tank, filled through a control valve and drained by gravity, so the outflow follows the
square root of the level and the downstream demand. The level transmitter reads through a transport
delay and adds noise; the valve actuator has its own lag. A PID block (sample time 0.5 s, ideal form,
percent of span) drives the valve. Tank size, valve size, delays and noise come from the seed and are
not told to the model.

The model gets these tools:

| Tool | What it does |
|---|---|
| `set_manual(valve_percent)` | controller to manual, valve output by hand |
| `set_auto(setpoint_m)` | controller to automatic (or a new setpoint) |
| `set_tuning(kc, ti_s, td_s, pv_filter_s, anti_windup)` | change any controller parameter |
| `set_outlet_demand(percent)` | ask operations to change the draw, 30 to 80 % |
| `wait(seconds, sample_s)` | let the plant run and return the trend |
| `submit()` | hand over for the acceptance test |

It has four hours of plant time and 40 tool calls. Plant time moves only when the model lets it run,
so thinking costs wall time, not plant time. Reaching the high-high (spill) or low-low (dry pump) alarm
while commissioning counts against it.

The acceptance test runs with the tuning the model left: the tank starts low, the controller switches
to automatic with setpoint 1.20 m, then the outlet demand steps up and down. The requirements, given
in the prompt, are: no alarm, inside 1.20 ± 0.03 m within a time limit and staying there, a maximum
overshoot, recovery into the band within a time limit after each demand step, and a limit on total
valve travel (a noisy derivative term or an absurd gain makes the valve chatter).

**Difficulty rule.** The limits are the worst result over a family of sensible tunings run through
the same test on the true plant: SIMC PI (tc equal to the effective delay) with anti-windup, with Kc
and Ti each scaled by 0.7, 1 and 1.4, plus 10 % slack (50 % on valve travel). So any PI tuning within
about 40 % of SIMC passes once anti-windup is on.

**Scripted agents**, checked by the test suite on 20 seeds:

* `reference` works only through the tools: it brings the level near the setpoint by hand, runs a 5 %
  valve step test, fits first order plus dead time (two points, 10 % and 63 %), tunes SIMC PI with
  anti-windup, checks the demand steps and submits. It passes every seed.
* `factory` submits the factory tuning without commissioning. It fails every seed.
* `aggressive` runs the same step test but tunes Ziegler-Nichols PI without anti-windup. The start-up
  saturates the valve, the integral winds up and the level overshoots by 30 to 40 cm. It fails every seed.

### `dsp.scale_readout`: make a weighing scale read right

A load cell under a 0..50 kg platform feeds an ADC (3.2 to 6.4 kHz, the only number the model is
told). The raw signal carries the load plus mains hum with its 3rd and 5th harmonics (near 50 or 60 Hz,
possibly a few tenths of a hertz off, drifting slightly), a strong tone from a switching converter
somewhere between 0.28 and 0.46 of the sample rate, and white noise. The hum alone is 2 to 5 kg.

The model builds the processing chain from blocks and has a budget of 12 multiply-adds per ADC sample:

| Block | Cost per sample at its input rate |
|---|---|
| `moving_average(n)` | 2 (running sum) |
| `notch(f0_hz, q)` | 5 |
| `lowpass(kind, order, fc_hz)`: Butterworth, Bessel or Chebyshev I | 5 per second-order section |
| `fir_lowpass(taps, fc_hz, window)` | taps |
| `decimate(factor)`: drops samples, no filtering of its own | 0; later blocks cost 1/factor |

Its tools are `set_chain`, a `spectrum` analyzer and a `scope` on the ADC signal or the chain output,
calibration weights (`set_load`), `run` to watch the display (10 readings per second, each the newest
chain output sample) and `submit`. It has 20 minutes of signal time and 40 tool calls.

The acceptance test starts the chain from rest, puts one weight on, swaps it for another, empties the
platform, and grades the display: settling into ±0.05 kg after each change within a time limit, and
over the last 2 s of each plateau a limit on the standard deviation and ±0.01 kg on the mean.

**Difficulty rule.** The limits are the worst result over a family of textbook designs: two moving
averages over one mains period, decimation by the same length, then a second-order Butterworth lowpass
at a tenth of the decimated rate, its cut-off scaled by 0.7, 1 and 1.4; 15 % slack on settling, 20 % on
noise. A second-order lowpass alone cannot meet noise and settling together, and decimating before
filtering folds the hum and the tone into the reading. Display sampling at 10 Hz turns any 50 Hz
residue into a slowly wandering offset, so it fails the accuracy limit too.

**Scripted agents**, checked on 20 seeds: `reference` measures the spectrum, finds the mains frequency,
builds that textbook chain, checks it with a 20 kg weight and passes every seed. `no_chain`,
`lowpass` (a 5 Hz second-order Butterworth only) and `decimate_first` (decimate to 50 Hz, then filter)
fail every seed. Sensible alternatives also pass, for example a notch on the mains fundamental followed
by a lowpass at the right cut-off.

### Running a model on a plant

```sh
uv run margin-bench play process.tank_level --agent reference --watch  # scripted engineer, free
uv run margin-bench play dsp.scale_readout --agent reference --watch
uv run margin-bench play --model anthropic/claude-sonnet-5.5 --effort high          # cost estimate only
uv run margin-bench play --model anthropic/claude-sonnet-5.5 --effort high --execute --watch
uv run margin-bench play --model qwen3:32b --base-url http://localhost:11434/v1 --watch   # any OpenAI-compatible server
```

`--watch` opens the live viewer in the browser: the plant animation, the trend, and every model turn
and tool call as it happens, with wall time, tokens and cost. Through OpenRouter, nothing is called
without `--execute`; the run prints an estimate first and stops at `--max-cost` (default $3). An
episode is written to `results/play/` after every step. API errors end an episode without a score;
they never count against the model.

```sh
uv run margin-bench export-play results/play/<episode>.json   # publish it to site/data/play
uv run margin-bench serve                                     # http://localhost:8765/play.html
```

| Model (OpenRouter list prices, 2026-10-15) | Typical episode (20 turns) | At the 60-turn limit |
|---|---|---|
| `anthropic/claude-sonnet-5.5` | about $1.47 | about $10.41 (stopped at $3) |
| `openai/gpt-6.1-sol` | about $1.47 | about $10.41 (stopped at $3) |
| `deepseek/deepseek-v4.1-flash` | about $0.04 | about $0.20 |

## Design tasks

### How a design task is graded

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

### Task families

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

### Running models on the design tasks

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
  play/          playground: environments, the tool-calling loop, episodes, local viewer
  tasks/         task families (generator, grader, reference, negative), by domain
  grading/       answer parsing and validation, continuous scoring, plot data format
  sims/          ngspice through its shared library
  harness/       OpenRouter client, run configs, attempt store, cost estimate
  report/        aggregation with bootstrap intervals, export for the site
tests/
runs/            run configs
results/         raw attempts (JSONL) and playground episodes
site/            static site and its data (play.html is the playground)
private/         hidden test set seeds and results (gitignored)
```

## License

MIT, see [LICENSE](LICENSE). It covers the code, the task generators, the run configs and the
published results.
