"""One registered decoder-gain sensitivity round on existing calibration data.

For each method, perturb the early/middle/late eight-layer group by +/-0.1
around its previously locked gain. Select on calibration only, then validate
the chosen physical manifest. Optimizer states and checkpoint masks stay fixed.
"""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-calibration", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--task-python", type=Path, required=True)
    ap.add_argument("--task-data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from experiments.lib.calibrated_archive import load_payload
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from run_residual_sweep_small import REGIMES, load_regime
    from run_system_bytes_comparison import excp_dequant
    cal = json.loads(args.source_calibration.read_text())
    source_record = Path(cal["source_record"])
    if not source_record.is_absolute():
        source_record = next(p.parent for p in args.source_calibration.resolve().parents if p.name == "results") / source_record
    assert sha256_file(source_record) == cal["source_record_sha256"]
    source = json.loads(source_record.read_text())
    archive = source_record.parent / source["archive"]["file"]
    assert sha256_file(archive) == cal["identity"]["source_archive_sha256"]
    source_manifest = json.loads(archive.read_text())
    assert source_manifest["format"] == "checkpoint-components-v1" and "gain" not in source_manifest
    args.out_dir.mkdir(parents=True, exist_ok=True)
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset+16])
    gate = [pool[i] for i in range(len(pool)) if i not in used][:64]
    assert batch_hash(gate) == cal["identity"]["gate_sha256"]
    baseline_gate = evaluate_lm(s.model, gate, s.device)
    baseline_validation = ctx["uncompressed"]
    from transformers import GPTNeoXForCausalLM
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = {n:p.detach().cpu().clone() for n,p in base_model.named_parameters()}
    payload = load_payload(archive)
    assert payload["eligible_parameter_names"] == s.names
    residual = {n:excp_dequant(v,base[n].shape,"cpu").float() for n,v in payload["model"].items()}
    buffers = payload["buffers"]
    del payload
    method, gain = cal["method"], cal["selected"]["gain"]
    cases = [("base", [gain]*3)]
    for group,name in enumerate(("early", "middle", "late")):
        for amount,suffix in ((-.1,"low"),(.1,"high")):
            gains = [gain]*3
            gains[group] = round(gain+amount,2)
            cases.append((name+"_"+suffix,gains))
    components = {}
    for key in ("state", "weights"):
        item = source_manifest[key]
        original = (archive.parent/item["file"]).resolve()
        assert original.parent == archive.parent.resolve() and sha256_file(original) == item["sha256"]
        components[key] = original
    identity = {"checkpoint_sha256":sha256_file(args.checkpoint), "source_calibration_sha256":sha256_file(args.source_calibration),
                "source_archive_sha256":sha256_file(archive), "gate_sha256":batch_hash(gate), "seed":42,
                "script_sha256":sha256_file(Path(__file__)), "cases":cases, "ppl_gate_percent":.5, "calibration_examples":128}
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    write_json(args.out_dir/"run-manifest.json",{"identity":identity,"source_calibration":str(args.source_calibration),"method":method,
               "code_commit":commit,"baseline_gate":baseline_gate,"selection":"highest calibration task mean within 0.5% PPL, then lower gate loss",
               "confirmation":"new 256 examples per task excluding every previously used example; old 1024 results retained"})
    points = {}
    for name,gains in cases:
        targets = {n:gains[int(n.split("layers.")[1].split(".")[0])//8] for n in s.names}
        with torch.no_grad():
            for n,p in s.model.named_parameters():
                p.copy_(base[n].to(s.device)+targets.get(n,1.)*residual[n].to(s.device))
        s.model.load_state_dict({**s.model.state_dict(),**buffers},strict=True)
        quality = evaluate_lm(s.model,gate,s.device)
        change = 100*math.expm1(quality["loss"]-baseline_gate["loss"])
        folder = args.out_dir/name
        folder.mkdir(exist_ok=True)
        manifest = {"format":"checkpoint-components-v1","parameter_gains":targets,"calibration_input_sha256":sha256_file(args.source_calibration)}
        component_bytes = 0
        for key in ("state","weights"):
            item = source_manifest[key]
            original = components[key]
            dest = folder/item["file"]
            if not dest.exists():os.link(original,dest)
            assert os.path.samefile(original,dest)
            manifest[key] = item
            component_bytes += dest.stat().st_size
        file = folder/"layer_gains.manifest.json"
        write_json(file,manifest)
        size = component_bytes+file.stat().st_size
        record = {"status":"completed","identity":{"checkpoint_sha256":identity["checkpoint_sha256"]},"method":method,"seed":42,
                  "sparsity":source["sparsity"],"config":source["config"],"gain_groups":gains,"case":name,
                  "gate":quality,"gate_ppl_change_percent":change,"passed_gate":change<=.5,"quantization":source["quantization"],
                  "archive":{"file":file.name,"bytes":size,"sha256":sha256_file(file),"original_checkpoint_bytes":args.checkpoint.stat().st_size},
                  "byte_compression":args.checkpoint.stat().st_size/size,"baseline_validation":baseline_validation,"code_commit":commit,
                  "source_record_sha256":sha256_file(source_record)}
        write_json(folder/"archive.json",record)
        write_json(folder/"completed.json",{"status":"completed","selected":{method:{"file":"archive.json","sha256":sha256_file(folder/"archive.json")}}})
        points[name] = record
        print(f"GATE {method} {name} PPL={change:.5f} accepted={change<=.5}",flush=True)
    del ctx,s,pool,base_model,base,residual,buffers
    torch.cuda.empty_cache()
    for name,point in points.items():
        if not point["passed_gate"]:continue
        if name == "base":
            assert abs(point["gate"]["loss"]-cal["selected"]["gate"]["loss"]) <= 1e-6
            point["task_accuracy"] = cal["selected"]["task_accuracy"]
            point["task_mean"] = cal["selected"]["task_mean"]
            point["tasks_reused_from_sha256"] = sha256_file(args.source_calibration)
        else:
            folder = args.out_dir/name
            command = [str(args.task_python),str(ROOT/"experiments/scripts/run_v100_downstream_subset.py"),"--checkpoint",str(args.checkpoint),
                       "--model-root",str(args.data_root/"models/pythia-410m"),"--encoded-dir",str(folder),"--out-dir",str(folder/"tasks"),
                       "--methods",method,"--task-data",str(args.task_data),"--subset","calibration","--examples","128"]
            subprocess.run(command,check=True)
            values = []
            for task,key in [("hellaswag","acc_norm,none"),("arc_easy","acc_norm,none"),("piqa","acc,none")]:
                row = json.loads((folder/"tasks"/(method+"_"+task+".json")).read_text())
                assert len(row["samples"]) == 128
                values.append(row["metrics"][key])
            point["task_accuracy"] = values
            point["task_mean"] = sum(values)/3
        print(f"TASKS {method} {name} mean={point['task_mean']:.6f}",flush=True)
    best = max((p for p in points.values() if p["passed_gate"]),key=lambda p:(p["task_mean"],-p["gate"]["loss"]))
    # Validate the chosen physical file after all calibration decisions are fixed.
    ctx = load_regime("R1",args.checkpoint.parent,args.data_root,"cuda:0")
    s = ctx["sweep"]
    chosen = args.out_dir/best["case"]
    payload = load_payload(chosen/best["archive"]["file"])
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root/"models/pythia-410m",dtype=torch.float32,local_files_only=True)
    base = dict(base_model.named_parameters())
    params = dict(s.model.named_parameters())
    with torch.no_grad():
        for n,value in payload["model"].items():
            params[n].copy_(base[n].to(s.device)+excp_dequant(value,base[n].shape,s.device).float())
    s.model.load_state_dict({**s.model.state_dict(),**payload["buffers"]},strict=True)
    actual_gate = evaluate_lm(s.model,gate,s.device)
    assert abs(actual_gate["loss"]-best["gate"]["loss"]) <= 1e-6
    best["validation"] = evaluate_lm(s.model,ctx["eval_batches"],s.device)
    best["validation_ppl_change_percent"] = 100*math.expm1(best["validation"]["loss"]-ctx["uncompressed"]["val"])
    write_json(chosen/"archive.json",best)
    write_json(chosen/"completed.json",{"status":"completed","selected":{method:{"file":"archive.json","sha256":sha256_file(chosen/"archive.json")}}})
    write_json(args.out_dir/"completed.json",{"status":"completed","identity":identity,"method":method,"points":points,"selected":best,
               "chosen_encoded_dir":str(chosen.resolve()),"code_commit":commit})
    print("SELECTED",method,best["case"],best["task_mean"],best["validation_ppl_change_percent"],flush=True)


if __name__ == "__main__":
    main()
