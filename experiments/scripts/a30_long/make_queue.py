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
            "--clip 1.0 --batch-size 4 --accum 16 --seq 2048 --eval-batches 40 --eval-batch-size 4 "
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


def b_main(method, decision=True, after_decision=False):
    tid = f"B_s42_{method}"
    dep = FINEWEB + (",runs/B_pilot_decide/DONE" if decision or after_decision else "")
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
    tasks.append(b_main("restart", decision=False, after_decision=True))
    write(tasks)


def write(tasks):
    out = sys.stdout
    out.write("# id\tdeps\tcommand\n")
    for tid, dep, cmd in tasks:
        out.write(f"{tid}\t{dep}\t{cmd}\n")


# ExCP protocol (2026-11-22): checkpoints compressed against the previous reconstructed checkpoint,
# training resumed from the reconstruction every fifth checkpoint.  The ExCP rule is the official
# per-tensor threshold at its cap, median(|d|) * 2.5 (alpha large), and DACP-G deletes as many entries.
#
# Both rules delete the same fraction at every checkpoint: the rate the official threshold reaches at its
# cap (median(|d|) * 2.5), measured as 0.905-0.917 on study A (mean 0.91) and 0.88 on study B.  The ExCP
# rule keeps the shape of the official threshold (per-tensor c * median(|d|), c set by the rate).
CHAIN_RATE = {"A": 0.91, "B": 0.88}


def chain_methods(study):
    p = CHAIN_RATE[study]
    return (("nocomp", "--method nocomp"),
            ("excp", f"--method excp_official --sparsity {p}"),
            ("dacp", f"--method dacp --sparsity {p} --kappa 1"))
A_CHAIN = A_COMMON.replace(f" --fail-steps {A_FAILS}", "") + " --ref-mode chain --ckpt-every 200 --break-every 1000"
B_CHAIN = (B_COMMON.replace("--ref-every 750 ", "--ref-every 0 ") +
           " --ref-mode chain --ckpt-every 250 --break-every 1250 --eval-every 250 "
           "--eval-after 10,25,50,100,200 --save-every 250")


def chain_tasks():
    tasks = []

    def a(seed):
        for name, extra in chain_methods("A"):
            tid = f"AC_s{seed}_{name}"
            tasks.append((tid, ALPACA, f"{RUN} {A_CHAIN} --seed {seed} {extra} --out $RUNS/{tid}"))

    a(42)
    # noise control: 0.1% random deletions, so later divergence from no compression measures training sensitivity
    tasks.append(("AC_s42_noise", ALPACA, f"{RUN} {A_CHAIN} --seed 42 --method random --sparsity 0.001 "
                                          "--out $RUNS/AC_s42_noise"))
    for name, extra in chain_methods("B"):
        tid = f"BC_s42_{name}"
        tasks.append((tid, FINEWEB, f"{RUN} {B_CHAIN} {extra} --out $RUNS/{tid}"))
    # resume pilot of study B (2026-11-22): at the first resumption DACP-G (kappa 1) had a quarter of the ExCP
    # damage but trailed it from ten steps later on.  From the no-compression state at step 3500 every variant
    # compresses the residual of the next interval, resumes at the break step 3750 and trains 200 steps
    # (deterministic kernels); pilot_decide_chain.py picks the DACP-G variant with the smallest mean gap after it
    pilot = (f"{RUN} {B_CHAIN} --init-state $RUNS/BK_states/state_3500.pt --ref-state $RUNS/BK_states/state_3500.pt "
             "--stop-at 3950 --deterministic")
    pb = CHAIN_RATE["B"]
    variants = [("nocomp", "--method nocomp"), ("excp", f"--method excp_official --sparsity {pb}"),
                ("noise", "--method random --sparsity 0.001"),
                ("dacp_k1", f"--method dacp --sparsity {pb} --kappa 1"),
                ("dacp_k4", f"--method dacp --sparsity {pb} --kappa 4"),
                ("dacp_k16", f"--method dacp --sparsity {pb} --kappa 16"),
                ("dacp_k1_q0.06", f"--method dacp --sparsity {pb} --kappa 1 --protect-top 0.06")]
    for name, extra in variants:
        tasks.append((f"BK/{name}", "runs/BK_states/state_3500.pt", f"{pilot} {extra} --out $RUNS/BK/{name}"))
    tasks.append(("BK_decide", ",".join(f"runs/BK/{n}/DONE" for n, _ in variants),
                  f"mkdir -p $RUNS/BK_decide && $PY $CODE/pilot_decide_chain.py --pilot $RUNS/BK --sparsity {pb} "
                  "--out $RUNS/BK_decide/decision.json > $RUNS/BK_decide/decide.log && date > $RUNS/BK_decide/DONE"))
    dec = "runs/BK_decide/DONE"
    # study B at the official ExCP default (alpha = 5e-5): the official threshold sets the deletion count at
    # every checkpoint (fewer deletions while the learning rate is large), and DACP-G deletes exactly as many
    # entries as the ExCP run did at the same step; the no-compression run is BC_s42_nocomp
    tasks.append(("BO_s42_excp", FINEWEB, f"{RUN} {B_CHAIN} --method excp_official --excp-alpha 5e-5 "
                                          "--out $RUNS/BO_s42_excp"))
    # the pilot-chosen DACP-G on both rate settings of study B
    tasks.append(("BC_s42_dacpt", f"{FINEWEB},{dec}", f"{RUN} {B_CHAIN} --method dacp "
                  "--decision $RUNS/BK_decide/decision.json --out $RUNS/BC_s42_dacpt"))
    tasks.append(("BO_s42_dacpt", f"{FINEWEB},{dec}", f"{RUN} {B_CHAIN} --method dacp "
                  "--decision $RUNS/BK_decide/decision.json --count-from $RUNS/BO_s42_excp/metrics.jsonl "
                  "--out $RUNS/BO_s42_dacpt"))
    a(43)
    a(44)
    return tasks


if __name__ == "__main__":
    if sys.argv[1:] == ["chain"]:
        write(chain_tasks())
    else:
        main()
