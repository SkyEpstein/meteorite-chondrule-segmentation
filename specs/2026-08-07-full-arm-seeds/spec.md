# Give the `full` arm its missing seeds (condensed spec)

Date: 2026-08-07. Decided by Sky this session, from a four-option MCQ.

## Problem

`src/run_cascade_arms.py` compares five stage-3 arms. Four of them (`control`,
`cleaned`, `cascade`, `fp_penalty`) have three seeds at 7,200 steps. The `full`
arm, which is the composition Sky believes in and the only arm that combines the
cleaned channels with the classical probability, has **one** seed at 7,200 steps
plus one at 30,000.

That is not just thin, it produced a wrong published number.
`logs/cascade_arms.log` reports

```
  full          0.5203   0.0158     0.2574   0.0055
    full        +0.1179   2-sigma bar 0.0448   RESOLVED
```

under a header reading "3 seeds each, 7,200 steps". Verified this session:
0.5203 is exactly the mean of the 7,200-step run (val 0.5092) and the
30,000-step run (val 0.5315), pooled across two incomparable budgets, n=2, and
the 0.0158 "SD" is the spread between those two budgets rather than seed
scatter. `save()` was later fixed to group by (arm, budget) and
`results/cascade_arms.json` is correct, but the printed summary and its RESOLVED
verdict were never regenerated.

## Approach

Run the two missing seeds at the matched budget, nothing else:

```
python -m src.run_cascade_arms --arm=full --seed=1,2
```

24 epochs x 300 steps = 7,200, identical to every other arm. The script is
resumable and skips completed (arm, seed, steps) triples, so seed 0 and the
30,000-step run are untouched. One process, because `evaluate.load` pulls train,
val and test.

No code change. The grouping defect is already fixed in `save()`; this run makes
the summary it prints defensible.

## Acceptance

1. `results/cascade_arms.json` holds a `("full", 7200)` summary row with
   `n_seeds = 3` and a real `val_iou_sd` / `test_iou_sd` over seeds.
2. The regenerated against-the-control verdict for `full` is computed from
   three 7,200-step seeds only, with the 30,000-step run in its own row.
3. The `full` vs `control` val gap is restated at n=3 and its RESOLVED or
   inside-the-noise verdict is reported honestly, whichever way it lands.
4. The test gap is reported next to it under the same 2-sigma rule. At n=3 the
   other arms' test bars run 0.058 to 0.068, so a test verdict of "inside the
   noise" is the expected outcome and is not a failure of the run.

## What this does NOT settle

Budget. The one 30,000-step `full` run (val 0.5315, test 0.2613) stays n=1, and
it is not verified as hyperparameter-matched to `unet2d_s30000` (test 0.29409,
recipe `phase1:p1_ctrl_b32`), so those two must not be differenced yet.

## RESULT (2026-08-07, run complete)

Seeds 1 and 2 landed. `full` at 7,200 steps is now n=3.

```
   seed 0   val 0.5092   test 0.2535   thr 0.90
   seed 1   val 0.4778   test 0.2399   thr 0.95
   seed 2   val 0.5089   test 0.2493   thr 0.95
```

|  | val | test |
|---|---|---|
| full, n=3, corrected | 0.4986 +/- 0.0181 | 0.2476 +/- 0.0070 |
| what was published (pooled, n=2 across budgets) | 0.5203 +/- 0.0158 | 0.2574 +/- 0.0055 |
| movement | -0.0217 | -0.0098 |

Against the control, same 2-sigma rule on both axes:

| arm | val gap | bar | verdict | test gap | bar | verdict |
|---|---|---|---|---|---|---|
| cleaned | +0.0536 | 0.0374 | RESOLVED | +0.0041 | 0.0612 | inside the noise |
| cascade | +0.0733 | 0.0397 | RESOLVED | -0.0026 | 0.0584 | inside the noise |
| full | +0.0962 | 0.0482 | RESOLVED | -0.0017 | 0.0595 | inside the noise |
| fp_penalty | -0.0150 | 0.0464 | inside the noise | -0.0460 | 0.0679 | inside the noise |

All four acceptance criteria met. The val conclusion SURVIVES the correction:
the composition is still resolved against the control at twice its bar, and the
ordering cleaned < cascade < full is intact, so the two halves do stack. The
test conclusion is that no input arm is separable from the control, and `full`'s
test gap changed sign against the n=1 estimate (+0.0042 to -0.0017).

## Second defect found in the same printout, NOT fixed

```
    full        +0.1291   2-sigma bar nan   inside the noise
```

That row is the 30,000-step run. Two problems. Its bar is `nan` because n=1 has
no SD, and `abs(dv) > nan` is False, so the code falls through to printing
"inside the noise", which is a substantive claim where the truth is "no verdict
possible at n=1". And it differences a 30,000-step arm against a 7,200-step
control, the cross-budget comparison this module's own regime string forbids.
Same class of bug as the pooling defect: a number that is not a result being
rendered as one.

## Bearing on the val-versus-test bar

`results/val_vs_test.json` puts the smallest val gap that predicts anything at
0.137 to 0.172 IoU. Every arm here has a val gap below that bar, and every one
of them produced a test gap inside the noise, including the one that is
comfortably resolved on validation. The 2026-08-04 handoff noted that only TWO
effects had both a val and a test number recorded. These four are seed-replicated
and land the same way, which strengthens that finding rather than adding to the
sign-flip tally.
