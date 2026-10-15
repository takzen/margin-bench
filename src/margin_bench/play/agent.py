"""Run a model on an environment: the tool-calling loop, its limits and a cost estimate."""

from __future__ import annotations

import json
from typing import Any

from margin_bench.harness.client import ApiError, ChatClient, Settings
from margin_bench.play.env import Environment
from margin_bench.play.episode import Episode

SYSTEM = """\
You operate a simulated plant through the tools you are given. Every tool call acts on the plant at once, and \
plant time only moves when a tool lets it run. Before each action, say briefly what you observed and what you \
are about to do and why. When the plant is ready, call submit. You cannot ask questions; nobody will answer."""

NUDGE = "Continue by calling the tools. Call submit when the plant is ready for the acceptance test."
MAX_TURNS = 60  # model turns, including ones without tool calls
MAX_IDLE_TURNS = 2  # turns in a row without a tool call before the run stops


def run_model(
    env: Environment,
    client: ChatClient,
    settings: Settings,
    episode: Episode,
    *,
    max_cost_usd: float | None = None,
    max_turns: int = MAX_TURNS,
) -> None:
    """Let the model work until it submits or a limit is hit, then run the acceptance test."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": env.prompt()},
    ]
    tools = env.tools()
    idle = 0
    stop = "turn limit"
    try:
        for _ in range(max_turns):
            turn = client.chat(messages, tools, settings)
            episode.model_turn(
                turn.text,
                reasoning=turn.reasoning,
                tool_calls=turn.tool_calls,
                latency_s=turn.latency_s,
                prompt_tokens=turn.prompt_tokens,
                completion_tokens=turn.completion_tokens,
                reasoning_tokens=turn.reasoning_tokens,
                cost_usd=turn.cost_usd,
                finish_reason=turn.finish_reason,
            )
            messages.append(turn.message)
            if not turn.tool_calls:
                idle += 1
                if idle > MAX_IDLE_TURNS:
                    stop = "stopped calling tools"
                    break
                messages.append({"role": "user", "content": NUDGE})
                continue
            idle = 0
            for call in turn.tool_calls:
                if env.submitted:
                    result: dict[str, Any] = {"error": "already submitted"}
                else:
                    try:
                        args = json.loads(call["arguments"] or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                    except ValueError as e:
                        result = episode.bad_call(call["name"], call["arguments"], f"could not parse the arguments: {e}")
                    else:
                        result = episode.tool_call(call["name"], args)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            if env.submitted:
                stop = "submitted"
                break
            if max_cost_usd is not None and episode.cost_usd >= max_cost_usd:
                stop = "cost limit"
                break
    except ApiError as e:
        # Not the model's fault: the episode is marked as an error and gets no score.
        episode.finish(None, stop_reason="api error", error=str(e))
        return
    episode.finish(env.finish(), stop_reason=stop)


def estimate_cost(prompt_price: float, completion_price: float, turns: int, *, base_tokens: int = 3000,
                  tokens_per_turn: int = 2500, output_per_turn: int = 2000) -> float:
    """Rough cost of one episode: the conversation grows by about one tool result per turn."""
    prompt_tokens = sum(base_tokens + i * tokens_per_turn for i in range(turns))
    return prompt_tokens * prompt_price + turns * output_per_turn * completion_price
