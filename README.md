<p align="center"><img src="assets/banner.svg" alt="margin-bench: can a language model run a plant?" width="100%"></p>

# margin-bench

Can a language model run a plant?

margin-bench gives a model a simulated plant and the tools an engineer would have: it can drive a valve,
tune a controller, build a signal chain, look at trends, spectra and scope traces, and put known weights
on a scale. It works on the plant as long as it needs, then hands over. An acceptance test on the plant
decides, by simulation only, never by another model.

Every run can be replayed on the site, step by step: the plant animation, every tool call and the data
it returned, the messages and reasoning the model shares, wall time, tokens and cost. A run can also be
watched live while the model works.

**Site:** https://takzen.github.io/margin-bench/

[![tests](https://github.com/takzen/margin-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/takzen/margin-bench/actions/workflows/tests.yml)

**Status:** two environments, `process.tank_level` and `dsp.scale_readout`. More plants are on the way:
an inline heater with a long transport delay, a DC motor, an inverted pendulum, a circuit with a fault.
No model has been run yet; the site shows the scripted agents.

## How an environment works

* **A plant, not a quiz.** Each environment is a simulated plant whose parameters come from a seed and
  are hidden from the model. The model only learns what it measures.
* **Tools.** The model acts through function calls. Plant time moves only when a tool lets it run, so
  thinking costs wall time, not plant time. Each environment has a budget of plant time and tool calls.
* **Acceptance test.** After `submit`, the environment runs a fixed test on a fresh copy of the plant
  with whatever the model left (a tuning, a signal chain) and checks written requirements, stated in
  the prompt. Each requirement gives a ratio that reaches 1 exactly when it is met; the score is the
  smallest one, so 1.000 means every requirement was met.
* **Each environment proves itself.** A scripted reference engineer, working only through the same
  tools, passes every seed, and scripted known-bad approaches fail every seed. The test suite checks
  both on 20 seeds.
* **Difficulty from design rules.** The limits of an acceptance test come from a family of textbook
  designs run on the true plant, never from any model's score.

## Environments

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

## Running a model

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                                                          # references pass, known-bad agents fail
uv run margin-bench envs
uv run margin-bench play process.tank_level --agent reference --watch  # scripted engineer, free
uv run margin-bench play dsp.scale_readout --model anthropic/claude-sonnet-5.5 --effort high   # cost estimate only
uv run margin-bench play dsp.scale_readout --model anthropic/claude-sonnet-5.5 --effort high --execute --watch
uv run margin-bench play process.tank_level --model qwen3:32b --base-url http://localhost:11434/v1 --watch
```

Models are called through [OpenRouter](https://openrouter.ai), or through any OpenAI-compatible server
with `--base-url` (Ollama, LM Studio, vLLM). For OpenRouter, put the key in `.env` (gitignored):

```
OPENROUTER_API_KEY=sk-or-...
```

`--watch` opens the live viewer in the browser. Through OpenRouter nothing is called without `--execute`;
the run prints an estimate first and stops at `--max-cost` (default $3). Every request sets OpenRouter's
`require_parameters`, so a provider that would silently ignore a setting is never used. An episode is
written to `results/play/` after every step. API errors end an episode without a score; they never count
against the model.

| Model (OpenRouter list prices, 2026-10-15) | Typical episode (20 turns) | At the 60-turn limit |
|---|---|---|
| `anthropic/claude-sonnet-5.5` | about $1.47 | about $10.41 (stopped at $3) |
| `openai/gpt-6.1-sol` | about $1.47 | about $10.41 (stopped at $3) |
| `deepseek/deepseek-v4.1-flash` | about $0.04 | about $0.20 |

```sh
uv run margin-bench export-play results/play/<episode>.json   # publish an episode to site/data/play
uv run margin-bench serve                                     # the site locally: http://localhost:8765/
```

The site in `site/` is plain HTML, JavaScript and SVG with no build step. GitHub Pages deploys it on
every push to `main`.

## Conflict of interest

This project is built with help from Claude, and Claude models are among those evaluated. To keep that
from tilting the results: difficulty comes from design rules, never from a model's score; environments
are never tuned to one model's results, Claude's least of all; every environment is approved by the
author, who works in these fields; and full transcripts are published with the results.

## Layout

```
src/margin_bench/
  play/          environments, the tool-calling loop, episode records, local viewer
  play/envs/     one module per plant: simulation, tools, acceptance test, scripted agents
  harness/       OpenRouter / OpenAI-compatible client, secrets from .env
  sims/          ngspice through its shared library (for the electronics plants)
  core.py        Score
  rng.py         seeded randomness that stays reproducible across versions
  scoring.py     requirement ratios
tests/
site/            the site: run list, replay and live viewer
results/play/    episode records
```

## License

MIT, see [LICENSE](LICENSE). It covers the code, the environments and the published runs.
