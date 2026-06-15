# autoresearch

This is an experiment to have an AI agent do its own streaming deep RL research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — the file you modify. Environment wrappers, actor/critic networks, ObGD optimizer, and the streaming training loop.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment streams through a fixed number of environment steps (`TOTAL_STEPS`, default 1,000,000) on a single environment, one transition at a time — no replay buffer, no batching. No GPU is required. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: the actor/critic architecture, the optimizer (ObGD or your own), hyperparameters (`LR`, `GAMMA`, `LAMBDA`, `ENTROPY_COEFF`, `KAPPA_POLICY`, `KAPPA_VALUE`, `HIDDEN_SIZE`), the observation/reward wrappers, the action distribution, etc.

**What you CANNOT do:**
- Change `ENV_NAME`, `SEED`, or `TOTAL_STEPS`. These are fixed so results are comparable across experiments.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change how the final summary (`mean_return`, etc.) is computed. That block is the ground-truth evaluation.
- Add a replay buffer, a target network, or batch updates. This is *streaming* RL: the agent learns from each transition once, in order, as it arrives — no storing and replaying past transitions, no separate lagged copy of the network, no minibatches.

**The goal is simple: get the highest `mean_return`** (mean episodic return over the final 5% of training — higher is better). Everything else is fair game: change the network architecture, the optimizer, the eligibility-trace/discount hyperparameters, the entropy bonus, the normalization wrappers. The only constraint is that the code runs without crashing and completes `TOTAL_STEPS` steps.

**RL runs are noisy.** A change only counts as an improvement if `mean_return` increases by more than `minimum_reward_improvement = 5` over the previous kept value. Smaller deltas (or regressions) are within the noise floor and are treated as "no improvement" — discard and reset.

**Compute** is a soft constraint. Since there's no fixed time budget, a change that meaningfully improves `mean_return` at the cost of somewhat higher `total_seconds` is fine; a change that dramatically increases per-step cost for a marginal gain is not worth it.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny `mean_return` improvement that adds 20 lines of hacky code? Probably not worth it. A tiny improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
mean_return:      1234.56
num_episodes:     842
total_steps:      1000000
total_seconds:    612.4
env:              HalfCheetah-v4
```

You can extract the key metric from the log file:

```
grep "^mean_return:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	mean_return	num_episodes	status	description
```

1. git commit hash (short, 7 chars)
2. `mean_return` achieved (e.g. 1234.56) — use 0.00 for crashes
3. `num_episodes` completed — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	mean_return	num_episodes	status	description
a1b2c3d	1234.56	842	keep	baseline
b2c3d4e	1450.10	790	keep	increase hidden size to 256
c3d4e5f	980.30	910	discard	switch to tanh activation
d4e5f6g	0.00	0	crash	NaN in policy std (exploding gradients)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^mean_return:\|^num_episodes:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If `mean_return` improved by more than `minimum_reward_improvement = 5` over the previous kept value, you "advance" the branch, keeping the git commit
9. Otherwise (improvement <= 5, including ties or regressions), you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Use your first (baseline) run to get a sense of how long `TOTAL_STEPS` takes on this machine — there is no fixed wall-clock budget, but a run that takes dramatically longer than the baseline (e.g. >3x) without a clear reason likely indicates a hang or a bug. Kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (NaN losses, shape mismatch, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment now takes longer than the original 5-minute LLM setup (it streams through `TOTAL_STEPS` environment steps), so pace your expectations accordingly — but the loop should still run for as many experiments as fit overnight. The user then wakes up to experimental results, all completed by you while they slept!
