# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real streaming deep RL setup and let it experiment autonomously overnight. It modifies the code, runs a training stream, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better agent. The training code here is a single-file implementation of Stream AC(&lambda;), the streaming actor-critic algorithm from [*Streaming Deep Reinforcement Learning Finally Works*](https://arxiv.org/abs/2410.14606) (Elsayed, Vasan & Mahmood, 2024) and its [reference implementation](https://github.com/mohmdelsayed/streaming-drl). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc.

## How it works

The repo is deliberately kept small and only really has two files that matter:

- **`train.py`** — environment wrappers (observation/reward normalization, episode-time feature), the actor and critic networks, the ObGD (eligibility-trace) optimizer, and the streaming training loop. Everything is fair game: network architecture, hyperparameters, optimizer, environment wrappers, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

Training streams through a **fixed number of environment steps** (`TOTAL_STEPS`, default 1,000,000), one transition at a time — no replay buffer, no batching, no GPU. The metric is **mean episodic return** over the final 5% of training — higher is better.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU required — this runs on CPU.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run a single training experiment (streams 1,000,000 env steps by default)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
train.py        — env wrappers, actor/critic networks, ObGD optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed step budget.** Training always streams through exactly `TOTAL_STEPS` environment steps (default 1,000,000), regardless of your specific platform. This makes experiments directly comparable regardless of what the agent changes (network architecture, optimizer, hyperparameters, etc).
- **Self-contained.** Dependencies are just PyTorch, NumPy, and Gymnasium (with MuJoCo and DeepMind Control Suite via shimmy). No replay buffer, no distributed training, no complex configs. One environment, one file, one metric.

## Algorithm credit

The streaming actor-critic agent (networks, sparse initialization, observation/reward normalization wrappers, and the ObGD optimizer) is ported from [mohmdelsayed/streaming-drl](https://github.com/mohmdelsayed/streaming-drl), specifically `stream_ac_continuous.py`. That repository is distributed under CC BY-NC 4.0 — see its `LICENSE.md` for terms governing the ported algorithm code.

```bibtex
@article{elsayed2024streaming,
  title={Streaming Deep Reinforcement Learning Finally Works},
  author={Elsayed, Mohamed and Vasan, Gautham and Mahmood, A Rupam},
  journal={arXiv preprint arXiv:2410.14606},
  year={2024}
}
```

## License

MIT (this repo's code, excluding the ported algorithm noted above)
