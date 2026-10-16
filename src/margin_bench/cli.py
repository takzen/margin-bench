"""Command line entry point."""

from __future__ import annotations

import argparse
import re
import time
import webbrowser
from datetime import datetime
from pathlib import Path

EPISODES_DIR = Path("results/play")
SITE_DIR = Path("site")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="margin-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("envs", help="list the environments and their scripted agents")

    pl = sub.add_parser("play", help="let a model (or a scripted agent) operate a simulated plant")
    pl.add_argument("env", help="environment id, see `margin-bench envs`")
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

    ex = sub.add_parser("export-play", help="publish finished episodes to the site")
    ex.add_argument("episodes", type=Path, nargs="+")
    ex.add_argument("--out", type=Path, default=SITE_DIR / "data" / "play")

    args = parser.parse_args(argv)

    if args.command == "envs":
        from margin_bench.play import ENVIRONMENTS, NEGATIVE, SCRIPTED

        for env_id, cls in sorted(ENVIRONMENTS.items()):
            bad = ", ".join(NEGATIVE[env_id])
            print(f"{env_id:22} {cls.title}  (scripted: reference passes; {bad} fail)")
    elif args.command == "play":
        _play(args)
    elif args.command == "serve":
        from margin_bench.play import serve

        serve.start(SITE_DIR, EPISODES_DIR, args.port)
        print(f"serving http://localhost:{args.port}/ (Ctrl+C to stop)")
        _wait_forever()
    elif args.command == "export-play":
        from margin_bench.play import serve

        rows = serve.export(args.episodes, args.out)
        print(f"{len(rows)} episode(s) in {args.out / 'index.json'}")


def _wait_forever() -> None:
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def _usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def _play(args: argparse.Namespace) -> None:
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
        serve.start(SITE_DIR, out.parent, args.port)
        url = f"http://localhost:{args.port}/?live=episodes/{out.name}"
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


if __name__ == "__main__":
    main()
