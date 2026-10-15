"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from margin_bench.tasks import FAMILIES, get_family


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="margin-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("families", help="list task families")

    p = sub.add_parser("prompt", help="print the prompt and spec of one task instance")
    p.add_argument("family")
    p.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("export-site", help="grade stored attempts again with plots and write site data")
    e.add_argument("results", type=Path, nargs="+", help="attempt JSONL files")
    e.add_argument("--out", type=Path, default=Path("site/data"))
    e.add_argument("--publish-hidden", action="store_true", help="allow results from private/ (reveals test instances)")

    dm = sub.add_parser("export-demo", help="write site data from reference and negative answers (no models)")
    dm.add_argument("--seeds", type=int, default=3, help="dev seeds per family")
    dm.add_argument("--out", type=Path, default=Path("site/data"))

    ts = sub.add_parser("new-test-set", help="draw secret seeds for a hidden test set into private/")
    ts.add_argument("name")
    ts.add_argument("--per-family", type=int, default=10)

    r = sub.add_parser("run", help="estimate the cost of a run config; call models only with --execute")
    r.add_argument("config", type=Path)
    r.add_argument("--execute", action="store_true", help="make paid API calls")

    rep = sub.add_parser("report", help="summarize an attempts JSONL file")
    rep.add_argument("results", type=Path)

    pl = sub.add_parser("play", help="let a model (or a scripted agent) operate a simulated plant")
    pl.add_argument("env", nargs="?", default="process.tank_level", help="environment id")
    pl.add_argument("--seed", type=int, default=0)
    who = pl.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="model id: an OpenRouter slug, or the server's model name with --base-url")
    who.add_argument("--agent", help="scripted agent instead of a model, e.g. reference")
    pl.add_argument("--base-url", help="any OpenAI-compatible server, e.g. http://localhost:11434/v1 (Ollama)")
    pl.add_argument("--api-key-env", help="environment variable holding the key for --base-url, if it needs one")
    pl.add_argument("--effort", help="reasoning effort, e.g. high")
    pl.add_argument("--max-tokens", type=int, default=16000, help="output tokens per model turn")
    pl.add_argument("--max-cost", type=float, default=3.0, help="stop the episode once this many USD are spent")
    pl.add_argument("--execute", action="store_true", help="make paid API calls (OpenRouter)")
    pl.add_argument("--watch", action="store_true", help="open the live viewer in the browser")
    pl.add_argument("--port", type=int, default=8765)
    pl.add_argument("--out", type=Path, help="episode file (default: results/play/...)")

    sv = sub.add_parser("serve", help="serve the site and results/play locally")
    sv.add_argument("--port", type=int, default=8765)

    ep = sub.add_parser("export-play", help="publish finished episodes to the site")
    ep.add_argument("episodes", type=Path, nargs="+")
    ep.add_argument("--out", type=Path, default=Path("site/data/play"))

    args = parser.parse_args(argv)

    if args.command == "families":
        for fid, fam in sorted(FAMILIES.items()):
            print(f"{fid}\t{fam.domain}")
    elif args.command == "prompt":
        task = get_family(args.family).generate(args.seed)
        sys.stdout.write(task.prompt)
        print(json.dumps(task.spec, indent=2))
    elif args.command == "export-site":
        _export_site(args.results, args.out, args.publish_hidden)
    elif args.command == "export-demo":
        from margin_bench.report import export

        summary, attempts = export.build(export.demo_records(range(args.seeds)), demo=True)
        export.write(args.out, summary, attempts)
        print(f"wrote {len(attempts)} demo attempts to {args.out}")
    elif args.command == "new-test-set":
        _new_test_set(args.name, args.per_family)
    elif args.command == "run":
        _run(args.config, args.execute)
    elif args.command == "report":
        _report(args.results)
    elif args.command == "play":
        _play(args)
    elif args.command == "serve":
        from margin_bench.play import serve

        serve.start(Path("site"), EPISODES_DIR, args.port)
        print(f"serving http://localhost:{args.port}/play.html (Ctrl+C to stop)")
        _wait_forever()
    elif args.command == "export-play":
        from margin_bench.play import serve

        rows = serve.export(args.episodes, args.out)
        print(f"{len(rows)} episode(s) in {args.out / 'index.json'}")


EPISODES_DIR = Path("results/play")


def _wait_forever() -> None:
    import time

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def _play(args: argparse.Namespace) -> None:
    import re
    import webbrowser
    from datetime import datetime

    from margin_bench.harness.client import Settings
    from margin_bench.harness.env import get_secret
    from margin_bench.harness.openrouter import BASE_URL, OpenRouterClient, fetch_models
    from margin_bench.play import SCRIPTED, agent, make, serve
    from margin_bench.play.episode import Episode

    env = make(args.env, args.seed)
    if args.agent:
        scripts = SCRIPTED.get(env.id, {})
        if args.agent not in scripts:
            raise SystemExit(f"no scripted agent {args.agent!r} for {env.id}; known: {', '.join(scripts)}")
        who = {"kind": "scripted", "name": args.agent, "settings": None}
        name = args.agent
    else:
        settings = Settings(model=args.model, reasoning_effort=args.effort, max_tokens=args.max_tokens)
        who = {"kind": "model", "name": args.model, "settings": settings.to_dict(),
               "endpoint": args.base_url or "openrouter"}
        name = args.model
        if args.base_url:
            api_key = get_secret(args.api_key_env) if args.api_key_env else None
        else:
            info = fetch_models().get(args.model)
            if info is None:
                raise SystemExit(f"{args.model!r} is not in OpenRouter's model list")
            p_in, p_out = (float(info["pricing"][k]) for k in ("prompt", "completion"))
            typical = agent.estimate_cost(p_in, p_out, 20)
            worst = agent.estimate_cost(p_in, p_out, agent.MAX_TURNS)
            print(f"{args.model}: about {_usd(typical)} for a typical episode (20 turns), "
                  f"{_usd(worst)} at the {agent.MAX_TURNS}-turn limit; the run stops at {_usd(args.max_cost)}")
            if not args.execute:
                print("dry run: no model was called. Add --execute to run.")
                return
            api_key = get_secret("OPENROUTER_API_KEY")
            if not api_key:
                raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env)")
        client = OpenRouterClient(api_key, base_url=args.base_url or BASE_URL)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    out = args.out or EPISODES_DIR / f"{env.id}-s{env.seed}-{slug}-{stamp}.json"
    episode = Episode(env, who, out)
    if args.watch:
        serve.start(Path("site"), out.parent, args.port)
        url = f"http://localhost:{args.port}/play.html?live=episodes/{out.name}"
        print(f"live view: {url}")
        webbrowser.open(url)

    if args.agent:
        SCRIPTED[env.id][args.agent](episode)
        episode.finish(env.finish(), stop_reason="submitted" if env.submitted else "script ended")
    else:
        agent.run_model(env, client, settings, episode, max_cost_usd=args.max_cost)

    d = episode.data
    r = d["result"] or {}
    u = d["usage"]
    verdict = "error: " + d["error"] if d["error"] else f"score {r.get('score', 0):.3f} {'PASS' if r.get('passed') else 'FAIL'}"
    print(f"{env.id} seed {env.seed}, {name}: {verdict} ({d['stop_reason']})")
    print(f"  wall {d['wall_s']:.1f} s, model {d['model_s']:.1f} s, {sum(s['kind'] == 'tool' for s in d['steps'])} tool calls, "
          f"{u['prompt_tokens'] + u['completion_tokens']} tokens, {_usd(u['cost_usd'])}")
    print(f"  episode: {out}")
    if args.watch:
        print("viewer still running (Ctrl+C to stop)")
        _wait_forever()


def _run(config: Path, execute: bool) -> None:
    from margin_bench.harness import runner
    from margin_bench.harness.env import get_secret
    from margin_bench.harness.openrouter import OpenRouterClient, fetch_models

    cfg = runner.RunConfig.from_toml(config)
    store = runner.AttemptStore(cfg.output)
    attempts = runner.plan(cfg)
    todo = runner.pending(attempts, store)
    print(f"run {cfg.name!r}: {len(attempts)} attempts planned, {len(attempts) - len(todo)} done, {len(todo)} pending")
    print(f"output: {cfg.output}")
    print(f"cost estimate assumes {cfg.expected_output_tokens} output tokens per call (worst case: the max_tokens cap)\n")

    rows = runner.estimate(cfg, todo, fetch_models())
    print(f"{'model settings':58} {'calls':>5} {'in $/M':>7} {'out $/M':>8} {'expected':>9} {'worst':>9}")
    for row in rows:
        p_in = "n/a" if row.usd_per_mtok_in is None else f"{row.usd_per_mtok_in:.2f}"
        p_out = "n/a" if row.usd_per_mtok_out is None else f"{row.usd_per_mtok_out:.2f}"
        print(f"{row.settings_key:58} {row.calls:>5} {p_in:>7} {p_out:>8} {_usd(row.expected_usd):>9} {_usd(row.worst_usd):>9}")
        for issue in row.issues:
            print(f"    ! {issue}")
    known = [r for r in rows if r.expected_usd is not None]
    print(f"\ntotal: expected {_usd(sum(r.expected_usd for r in known))}, worst {_usd(sum(r.worst_usd for r in known))}")
    if cfg.max_cost_usd is not None:
        print(f"budget cap: ${cfg.max_cost_usd:.2f} (no new calls once spent; already spent ${store.spent_usd():.4f})")

    if not execute:
        print("\ndry run: no model was called. Add --execute to run.")
        return
    if any(r.issues for r in rows):
        raise SystemExit("refusing to execute: fix the issues listed above first")
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env)")
    summary = runner.execute(cfg, todo, OpenRouterClient(api_key), store)
    print(f"\ndone: {summary}")


def _export_site(results: list[Path], out: Path, publish_hidden: bool) -> None:
    from margin_bench.harness.runner import AttemptStore
    from margin_bench.report import export

    hidden = [p for p in results if "private" in p.resolve().parts]
    if hidden and not publish_hidden:
        raise SystemExit(f"refusing to publish hidden test results {hidden}; pass --publish-hidden once the version is final")
    records = [r for p in results for r in AttemptStore(p).records()]
    summary, attempts = export.build(records)
    export.write(out, summary, attempts)
    print(f"wrote {len(attempts)} attempts from {len(results)} file(s) to {out}")


def _new_test_set(name: str, per_family: int) -> None:
    import secrets

    path = Path("private") / f"{name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a test set is never overwritten")
    # Above the dev range, drawn from the OS's secure source: not guessable from the code.
    seeds = {fid: sorted(10**6 + secrets.randbelow(2**31 - 10**6) for _ in range(per_family)) for fid in FAMILIES}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(seeds, indent=1), encoding="utf-8")
    print(f"wrote {per_family} secret seeds per family to {path}; it is gitignored, keep a backup outside the repo")


def _usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def _report(results: Path) -> None:
    from margin_bench.harness.runner import AttemptStore
    from margin_bench.report.aggregate import summarize

    rows = summarize(AttemptStore(results).records())
    print(f"{'model settings':58} {'family':22} {'n':>4} {'score [95% CI]':>21} {'pass [95% CI]':>21} {'fmt':>4} {'cost':>9}")
    for r in rows:
        lo, hi = r["mean_score_ci"]
        plo, phi = r["pass_rate_ci"]
        cost = "n/a" if r["total_cost_usd"] is None else f"${r['total_cost_usd']:.3f}"
        print(
            f"{r['settings_key']:58} {r['family_id']:22} {r['attempts']:>4} "
            f"{r['mean_score']:.3f} [{lo:.3f}, {hi:.3f}] {r['pass_rate']:.3f} [{plo:.3f}, {phi:.3f}] "
            f"{r['format_errors']:>4} {cost:>9}"
        )


if __name__ == "__main__":
    main()
