# 3-MINUTE DEMO — final order (lead with the non-generic moment)

One browser window. Tab 1: http://127.0.0.1:8008 · Tab 2: Weave traces · Tab 3: /about (backup).

## T-2 min
1. `make serve` running. Composer: poison OFF, --iterations 8 → Run (runs in background).
2. Sidebar → lab_report_run → pause → › ×3  → screen shows ROUND 3 (21% confidence override).

## 0:00 OPEN ON THE OVERRIDE (paused on round 3)
"Most agent loops improve a number. Ours knows when it's unsure. Look at this round: it scored
69%, two moves in a row had failed, and its decision-maker was only 21% sure what to do next.
Below 40%, the loop stops trusting itself and asks a reasoning model for a second opinion —
which overruled it. Next round: plus fifteen points. That's the whole project in one frame:
it acts, it measures, it knows when it doesn't know."

## 0:35 THE LOOP (› once; point at the flowchart)
"Train, take the same fixed exam every round, diagnose, pick one move. Four hospitals,
unlocked one at a time. It started at 55% — guessing 'everyone's sick' — with one small
hospital. The diagnosis reads its own Weave history and cites it (memory: Weave · cited N);
the move comes from TypeSafe's model in a third of a second, with a probability on every
option (five branches). Things that didn't work get taken off the menu."

## 1:10 THE POISON STORY (click poison_va → › ×2)
"Then we tried to fool it. We secretly flipped 40% of one hospital's labels. It unlocked
that hospital, the score dropped (red building, −3.8), and its confidence collapsed on the
next decision. It wrote in its own diagnosis that adding VA caused the drop — citing the
trace. Nobody told it. When it ran out of good moves, it stopped rather than pretend."

## 1:55 PROOF (tab 2 → newest diagnose call → output)
"None of this is a slide. Every round is a Weave trace — here are the trace IDs that
diagnosis cited. We replayed the same decisions through three decision-makers in a Weave
evaluation, including plain rules — honestly, on nine decisions they're within noise on
accuracy. What isn't noise: the structured model is twenty times faster and gives us a
confidence we can gate on."

## 2:30 THE LIVE RUN AS THE FINISH (tab 1 → refresh)
"This one has been running live since I started talking — (read the caption). Fifty-five to
eighty-seven percent in six rounds, on its own, past the published benchmark."

## 2:50 CLOSE
"Weave is its memory, W&B Inference its reasoning, TypeSafe its reflexes, marimo its
notebook, ARIA is who it asks for new code. Click any round and we'll show you why."

## Clicks: (pre) lab_report_run→pause→›×3 · (0:35) › · (1:10) poison_va→›×2 · (1:55) Weave diagnose · (2:30) tab 1 refresh

## If it breaks
Live run stalled → stay on lab_report_run ("the recorded run is the same loop").
Wi-Fi gone → replays work from disk; say Weave needs network; move on.
Hard question → "let me click into that round" — and do it.

## Q&A
- Gaming the metric? Fixed stratified exam from every site before round 0; never trained on.
- LLM vs rules? Within noise on n=9; the win is 20× latency + calibrated confidence for the gate.
- Stubbed? ARIA live fix not executed (protocol built); memory is project-wide (disclosed).
- Poisoned run missed target? Plateaued at 82.7% and stopped — the honest behaviour we want.
