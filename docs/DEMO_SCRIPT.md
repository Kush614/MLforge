# 3-MINUTE DEMO SCRIPT (strictly enforced — rehearse with a stopwatch)

## Prep (5 min before)
1. `make serve` → http://127.0.0.1:8008 full width (light theme).
2. Tab 2: Weave traces (wandb.ai/<entity>/agentforge-loop/weave). Tab 3 (optional): `make report`.
3. Sidebar → click `poison_va`, pause at round 3 so the screen is never empty.
4. Composer: `--iterations 8`, poison OFF. Don't press Run yet.
5. Have a clean run that beat the target in the Runs list as a fallback (run `make run` tonight).

## 0:00–0:20 — hook (replay paused on screen)
"The slow part of ML is a human staring at metrics, guessing: more data? different model?
We made that guess the agent's job — and made every guess observable."

## 0:20–1:40 — LIVE (press Run ↵). Narrate the caption, not the code.
- Round 0: "One tiny hospital, 55% — basically guessing 'everyone's sick'."
- First decision (five branches): "TypeSafe picks exactly one move with a probability on each
  option — 300 ms, no free text to parse."
- Hospital unlocks: "It chose to earn more data rather than swap models."
- Diagnose box: "Before it reasons it reads its own Weave history — those tags cite past traces."
- Score crosses the amber tick: "That's the published benchmark. Target beaten, it stops."
- Slow round filler: the ledger — "every move has a measured effect on the same held-out exam."
- HARD CAP 1:20 on this segment. Move on even mid-round; the fallback is a past run.

## 1:40–2:20 — the memory story (sidebar → poison_va, › to round 2–3)
"We secretly corrupted one hospital's labels. It added VA, the score fell, and its confidence
collapsed from 0.89 to 0.26 — below our floor — so it asked for a second opinion. It never
learned that from us. It learned it from its own ledger." Point at the red hospital + amber badge.

## 2:20–2:45 — proof (Weave tab). Click one diagnose call → cited_trace_ids.
"Every decision is one click deep — and we ran the same diagnoses through three decision-makers
in a Weave eval: TypeSafe, the reasoning model, and plain rules."

## 2:45–3:00 — close
"Weave is its memory, W&B Inference its reasoning, TypeSafe its reflexes, marimo its lab
notebook, and when it runs out of moves it writes a fix request for ARIA. Ask us about any round."

## Don't show
Code tab, 3D page, terminal, dry runs.

## Q&A
- Gaming the metric? Fixed stratified test split from every site, never trained on, same seed.
- LLM vs rules? Honest: on 9 diagnoses rules are competitive on Δacc (small n). Not noise:
  Jev ~20× faster with calibrated uncertainty, which we use as a gate.
- Stubbed? ARIA live fix not yet executed (protocol built); hosted MCP not wired (Weave client
  API used). Everything else ran live.
- Poisoned run missed target? It plateaued at 83% and stopped — the honest behaviour we want.
