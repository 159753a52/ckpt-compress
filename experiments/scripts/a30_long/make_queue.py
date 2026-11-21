#!/usr/bin/env python
"""Write the task queue of the A30 long-run study (tab-separated: id, deps, command).

Workers take the first task whose dependencies exist and that no live worker has
claimed, so the order below is the priority.  Commands expand $PY, $ROOT, $RUNS,
$DATA and $CODE from code/env.sh.
"""
from __future__ import annotations

import sys

A_FAILS = ",".join(str(300 + 600 * k) for k in range(10))             # 300, 900, ..., 5700
B_FAILS = ",".join(str(375 + 750 * k) for k in range(10))             # 375, 1125, ..., 7125

A_COMMON = ("--mode finetune --model EleutherAI/pythia-410m --tokenizer EleutherAI/pythia-410m "
            "--alpaca-parquet $DATA/alpaca/train.parquet --cache-dir $DATA/cache "
            "--total-steps 6000 --warmup 250 --lr 1e-5 --min-lr-ratio 0 --weight-decay 0.01 --beta2 0.999 "
            "--clip 0 --batch-size 4 --accum 4 --seq 512 --eval-every 250 --eval-after 10,25,50,100,200 "
            f"--eval-batch-size 8 --save-every 600 --fail-steps {A_FAILS} --stages 16 --score-batches 16 "
            "--score-batch-size 2 --kappa 1 --amp bf16 --deterministic")
B_COMMON = ("--mode pretrain --tokenizer EleutherAI/pythia-410m --token-dir $DATA/fineweb_edu "
            "--total-steps 7500 --warmup 75 --lr 6e-4 --min-lr-ratio 0.1 --weight-decay 0.1 --beta2 0.95 "
            "--clip 1.0 --batch-size 8 --accum 8 --seq 2048 --eval-batches 40 --eval-batch-size 4 "
            "--ref-every 750 --stages 16 --score-batches 16 --score-batch-size 2 --amp bf16 --seed 42 "
            "--hidden 768 --layers 12 --heads 12 --ffn 3072")
RUN = "$PY $CODE/long_run.py"
ALPACA = "data/alpaca/DONE"
FINEWEB = "data/fineweb_edu/DONE"
PILOT_T = {375: 0, 1125: 750}                                          # failure step -> reference step
PILOT_RATES = (0.5, 0.7, 0.84, 0.95)
PILOT_KAPPAS = (1, 3)


def a_tasks(seed, methods=(("nocomp", None), ("excp", 0.84), ("dacp", 0.84), ("excp", 0.95), ("dacp", 0.95))):
    out = []
    for m, p in methods:
        tid = f"A_s{seed}_{m}" + (f"_p{p}" if p else "")
        sp = f" --sparsity {p}" if p else ""
        out.append((tid, ALPACA, f"{RUN} {A_COMMON} --seed {seed} --method {m}{sp} --out $RUNS/{tid}"))
    return out


def b_main(method, decision=True):
    tid = f"B_s42_{method}"
    dep = FINEWEB + (",runs/B_pilot_decide/DONE" if decision else "")
    dec = " --decision $RUNS/B_pilot_decide/decision.json" if decision else ""
    return (tid, dep, f"{RUN} {B_COMMON} --method {method}{dec} --fail-steps {B_FAILS} --eval-every 250 "
                      f"--eval-after 10,25,50,100,200 --save-every 250 --out $RUNS/{tid}")


def pilot_tasks():
    pre = ("B_pilot_pre", FINEWEB,
           f"{RUN} {B_COMMON} --method nocomp --stop-at 1125 --keep-states 0,375,750,1125 --eval-every 125 "
           "--eval-after '' --save-every 125 --out $RUNS/B_pilot_pre")
    evals = []
    for t, r in PILOT_T.items():
        base = (f"{RUN} {B_COMMON} --init-state $RUNS/B_pilot_pre/state_{t}.pt "
                f"--ref-state $RUNS/B_pilot_pre/state_{r}.pt --fail-steps {t} --stop-at {t + 125} "
                f"--eval-every 125 --eval-after '' --save-every 0")
        variants = [("nocomp", ""), ("restart", "")]
        variants += [(f"excp_p{p}", f" --sparsity {p}") for p in PILOT_RATES]
        variants += [(f"dacp_p{p}_k{k}", f" --sparsity {p} --kappa {k}") for p in PILOT_RATES for k in PILOT_KAPPAS]
        for name, extra in variants:
            method = name.split("_")[0]
            tid = f"B_pilot/t{t}_{name}"
            evals.append((tid, "runs/B_pilot_pre/DONE", f"{base} --method {method}{extra} --out $RUNS/{tid}"))
    deps = ",".join(f"runs/{tid}/DONE" for tid, _, _ in evals)
    decide = ("B_pilot_decide", deps,
              "mkdir -p $RUNS/B_pilot_decide && $PY $CODE/pilot_decide.py --pilot $RUNS/B_pilot "
              "--out $RUNS/B_pilot_decide/decision.json && date > $RUNS/B_pilot_decide/DONE")
    return pre, evals, decide


def main():
    pre, evals, decide = pilot_tasks()
    tasks = []
    tasks += a_tasks(42)
    tasks.append(pre)
    tasks.append(b_main("nocomp", decision=False))
    tasks += evals
    tasks.append(decide)
    tasks.append(b_main("dacp"))
    tasks.append(b_main("excp"))
    tasks += a_tasks(43)
    tasks += a_tasks(44)
    tasks += a_tasks(42, methods=(("inshrink", 0.84),))
    tasks.append(b_main("restart", decision=False))
    out = sys.stdout
    out.write("# id\tdeps\tcommand\n")
    for tid, dep, cmd in tasks:
        out.write(f"{tid}\t{dep}\t{cmd}\n")


if __name__ == "__main__":
    main()
