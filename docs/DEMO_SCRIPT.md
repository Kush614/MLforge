# 3-MINUTE DEMO SCRIPT (strictly enforced — rehearse with a stopwatch)

0:00-0:20 — The hook. "The slow part of ML is a human staring at metrics guessing
what is wrong. We made that the agent's job." Show the loop diagram (one slide, max).

0:20-1:50 — LIVE loop (pre-warmed, mid-run). Screen split three ways:
 1. terminal: iteration running, accuracy climbing toward the benchmark line
 2. Weave UI: the diagnosis trace — point at the evidence tags AND the past-trace IDs
    it cited ("its memory is its own trace history")
 3. marimo lab report: the section it just wrote about itself
Narrate ONE decision end-to-end: "here it saw a 14-point train/test gap, called it
overfitting, and chose to acquire the Hungary site's data instead of switching models
— because its traces show model-switching did not help two iterations ago."

1:50-2:20 — The ARIA moment: show the fix request it emitted and the diff ARIA applied.
"When the problem is its own code, it delegates."

2:20-2:50 — The number: final accuracy vs published baseline, on the Weave eval chart.
Mention TypeSafe-vs-Inference comparison eval in one sentence.

2:50-3:00 — "Every decision you just saw is one click deep in Weave. Ask us about any
of them." Stop talking.

Q&A prep: how it avoids gaming the metric (seeded split, held-out data);
what breaks it (plateau detection); what is stubbed vs real (answer honestly).
