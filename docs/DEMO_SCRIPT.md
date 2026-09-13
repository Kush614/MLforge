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

---

## Word-for-word (≈390 words, 3:00 at a calm pace)

[T-90 s: composer → poison off, --iterations 8 → Run. Console tab up, poison_va visible.]

0:00 HOOK (live run on round 1–2)
"The slow part of machine learning isn't training. It's the human staring at metrics
afterwards, guessing — more data? different model? different features? — and trying again.
We made that guess the agent's job. And we made every guess observable."

0:20 THE LOOP (point at the flowchart as the dot moves)
"This is it running live right now. It started with one small hospital and a bad model —
55 percent, which is just guessing 'everyone's sick.' Every round it trains, takes the same
fixed exam, diagnoses what's wrong, and picks exactly one move."
(Diagnose) "The diagnosis is a reasoning model on W&B Inference — but before it reasons, it
reads its own history. Those tags come with the Weave trace IDs it cited. Its memory is its
own trace history."
(five branches) "The move is picked by TypeSafe's Jev — a model that doesn't write text, it
answers typed questions with a probability on every option. Three hundred milliseconds,
nothing to parse, and it can't produce an illegal move."
(when a Δ banner lands, read the caption) "Plus seventeen points — the last move helped.
That's on the same held-out exam every round, so it's a measured effect, not a story."

1:20 HARD CUT even mid-round → click poison_va, then › twice
"Here's the part I care about. In this run we secretly corrupted one hospital's labels.
Watch: it unlocked VA, the score fell almost four points — (red hospital) — and on the next
decision its confidence collapsed. When confidence drops below forty percent, the loop asks
the reasoning model for a second opinion. (amber badge) Nobody told it VA was bad. It learned
that from its own ledger — and when it ran out of good moves, it stopped, rather than pretend."

2:10 WEAVE TAB → newest diagnose call → output
"Every decision you've seen is one click deep in Weave — here are the trace IDs that
diagnosis cited. And we replayed the same diagnoses through three decision-makers in a Weave
evaluation: TypeSafe, the reasoning model, and plain rules. Honest answer: on nine decisions
they're within noise on accuracy — but Jev is twenty times faster and gives us a confidence
we can gate on."

2:40 CONSOLE → click lab_report_run
"On its own, this thing went from fifty-five to eighty-seven percent in six rounds — past the
target and past the published number — and wrote its own lab report about how."

2:50 CLOSE
"Weave is its memory. W&B Inference is its reasoning. TypeSafe is its reflexes. marimo is its
notebook. And when it runs out of moves, it writes a fix request for ARIA. Ask us about any
round." (Stop talking.)

If the live run stalls: click lab_report_run and keep narrating — the words work on the replay.
