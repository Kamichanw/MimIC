import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
if not os.path.isabs(RESULTS_DIR):
    RESULTS_DIR = str((ROOT_DIR / RESULTS_DIR).resolve())


def merge_args(base_args, new_args):
    base_dict = {arg.partition("=")[0]: arg for arg in base_args}

    if new_args:
        new_dict = {arg.partition("=")[0]: arg for arg in new_args}
        base_dict.update(new_dict)

    return list(base_dict.values())


def get_avail_devices(devices, requires_memory=None):
    if not hasattr(get_avail_devices, "cached_requires_memory"):
        if requires_memory is None:
            raise ValueError("requires_memory must be provided on the first call.")
        get_avail_devices.cached_requires_memory = requires_memory

    effective_requires_memory = (
        requires_memory if requires_memory is not None else get_avail_devices.cached_requires_memory
    )

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to query GPU memory.")

    free_memory = result.stdout.strip().split("\n") if result.stdout.strip() else []
    free_gpus = [
        str(idx)
        for idx, mem in enumerate(free_memory)
        if int(mem) > effective_requires_memory
    ]

    if devices:
        return ",".join(sorted(set(devices.split(",")) & set(free_gpus), key=int))
    return ",".join(free_gpus)


def run_train(runname, dataset, num_query_sample, num_shot, model_name, train_args, devices):
    process = subprocess.Popen(
        [sys.executable, str(ROOT_DIR / "src" / "train.py")]
        + merge_args(
            [
                f"runname={runname}",
                f"model_name={model_name}",
                f"data.num_query_samples={num_query_sample}",
                f"data.name={dataset}",
                f"data.num_shot={num_shot}",
            ],
            shlex.split(train_args),
        ),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": get_avail_devices(devices)},
        cwd=ROOT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    return process.wait() == 0


def run_eval(ckpt_path, dataset, num_query_sample, num_shot, model_name, eval_args, devices):
    visible_devices = get_avail_devices(devices)
    if not visible_devices:
        raise RuntimeError("No available GPUs found for evaluation.")

    num_processes = len(visible_devices.split(","))
    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
    ]
    if num_processes > 1:
        command.append("--multi_gpu")
    command.append(str(ROOT_DIR / "src" / "eval.py"))
    command += merge_args(
        [
            f"ckpt_path={ckpt_path or 'null'}",
            f"model_name={model_name}",
            f"data.name={dataset}",
            f"data.num_shot={num_shot}",
            f"data.num_query_samples={num_query_sample if num_query_sample is not None else 'null'}",
        ],
        shlex.split(eval_args),
    )

    process = subprocess.Popen(
        command,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_devices},
        cwd=ROOT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    if process.wait() != 0:
        target = ckpt_path or f"icl-{model_name}-{dataset}"
        print(f"Evaluation failed for {target}", file=sys.stderr)
        return False
    return True


def run_analyze(runname, dataset, num_query_sample, num_shot, model_name, analyze_args):
    if "icl" in runname:
        expand_runname = f"{runname}-{model_name}-{dataset}"
    else:
        expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"

    subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "analyze.py")]
        + merge_args(
            [
                f"model_name={model_name}",
                f"record_dir={os.path.join(RESULTS_DIR, 'record', expand_runname)}",
                f"data.name={dataset}",
                f"data.num_shot={num_shot}",
                f"data.num_query_samples={num_query_sample if num_query_sample is not None else 'null'}",
            ],
            shlex.split(analyze_args),
        ),
        env=os.environ,
        cwd=ROOT_DIR,
        check=True,
    )


def execute_eval(runname, tasks, model_name, eval_args, devices):
    for dataset, num_query_sample, num_shot in tasks:

        if "icl" in runname:
            print(f"Evaluating ICL task: dataset={dataset}, num_query_sample={num_query_sample}, num_shot={num_shot}")
            if not run_eval(None, dataset, num_query_sample, num_shot, model_name, eval_args, devices):
                return False
            continue

        expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
        ckpt_dir = os.path.join(RESULTS_DIR, "ckpt", expand_runname)
        if not os.path.isdir(ckpt_dir):
            print(f"Checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
            return False

        for epoch_ckpt in sorted(os.listdir(ckpt_dir), key=lambda name: int(re.findall(r"\d+", name)[0])):
            ckpt_path = os.path.join(ckpt_dir, epoch_ckpt)
            print(
                f"Evaluating checkpoint: dataset={dataset}, num_query_sample={num_query_sample}, num_shot={num_shot}, ckpt={epoch_ckpt}"
            )
            if not run_eval(ckpt_path, dataset, num_query_sample, num_shot, model_name, eval_args, devices):
                return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run training, distributed evaluation, and analysis tasks.")
    parser.add_argument("-r", "--runname", required=True, help="Name for the current run.")
    parser.add_argument("-d", "--datasets", required=True, help="Comma-separated list of datasets.")
    parser.add_argument("-m", "--model-name", required=True, help="Name of the model to use.")
    parser.add_argument("-q", "--num-query-samples", default="", help="Comma-separated list of query samples.")
    parser.add_argument("-s", "--num-shots", required=True, help="Comma-separated list of shots.")
    parser.add_argument("-t", "--train", action="store_true", help="Enable train mode.")
    parser.add_argument("-e", "--eval", action="store_true", help="Enable distributed eval mode. If set, analyze mode will also be enabled.")
    parser.add_argument("-a", "--analyze", action="store_true", help="Enable analyze mode.")
    parser.add_argument("--train-args", required=False, help="Additional training arguments.")
    parser.add_argument("--eval-args", required=False, help="Additional evaluation arguments.")
    parser.add_argument("--analyze-args", required=False, help="Additional analysis arguments.")
    parser.add_argument("--devices", required=False, help="Comma-separated list of device ids used to train or evaluate.")
    parser.add_argument("--wait-devices-timeout", type=int, default=0, help="Maximum time in minutes to wait for free GPUs. If <= 0, exit immediately if no adequate GPUs are available.")
    parser.add_argument("--requires_memory", type=int, default=20000, help="Minimum free CUDA memory in MB required to run train or eval.")
    parser.add_argument("--wait-n-devices", type=int, default=1, help="Minimum number of free devices required to start.")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    num_query_samples = args.num_query_samples.split(",") if args.num_query_samples else []
    num_shots = args.num_shots.split(",")
    runname = args.runname
    model_name = args.model_name
    train_args = getattr(args, "train_args") or ""
    eval_args = getattr(args, "eval_args") or ""
    analyze_args = getattr(args, "analyze_args") or ""
    devices = args.devices
    timeout = args.wait_devices_timeout
    min_devices = max(args.wait_n_devices, 1)
    requires_memory = args.requires_memory

    if not (args.train or args.eval or args.analyze):
        args.train = args.eval = args.analyze = True

    if (args.eval or args.train):
        available = get_avail_devices(devices, requires_memory)
        if (not available) or len(available.split(",")) < min_devices:
            print(f"Cannot find at least {min_devices} device(s). Start waiting...")
            for i in range(timeout):
                available = get_avail_devices(devices, requires_memory)
                if available and len(available.split(",")) >= min_devices:
                    break
                time.sleep(60)
                print(f"Waited for {min_devices} device(s) for {i + 1} mins")
            else:
                print(f"Cannot find at least {min_devices} device(s). Timeout, exit...")
                return

    if num_query_samples:
        tasks = list(product(datasets, num_query_samples, num_shots))
    else:
        if args.train:
            raise RuntimeError("The option -q/--num-query-samples is required if train mode is enabled.")
        tasks = list(product(datasets, [None], num_shots))

    if args.train:
        print("Starting training phase...")
        for dataset, num_query_sample, num_shot in tasks:
            if not run_train(
                runname,
                dataset,
                num_query_sample,
                num_shot,
                model_name,
                train_args,
                devices,
            ):
                return

    if args.eval:
        print("Starting evaluation phase...")
        if not execute_eval(runname, tasks, model_name, eval_args, devices):
            return

    if args.analyze:
        print("Starting analysis phase...")
        for dataset, num_query_sample, num_shot in tasks:
            run_analyze(runname, dataset, num_query_sample, num_shot, model_name, analyze_args)


if __name__ == "__main__":
    main()
