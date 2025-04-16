import torch
import torch_npu
import argparse
import os
import re
import subprocess
import sys
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shlex
import paths
import subprocess


def merge_args(base_args, new_args):
    base_dict = {arg.partition("=")[0]: arg for arg in base_args}

    if new_args:
        new_dict = {arg.partition("=")[0]: arg for arg in new_args}
        base_dict.update(new_dict)

    return list(base_dict.values())

def get_avail_devices(devices, requires_memory=None):
    # use npu-smi to get information about available devices
    result = subprocess.run(
        ["npu-smi", "info", "-l"],  
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run npu-smi: {result.stderr.strip()}")

    lines = result.stdout.strip().split("\n")
    available_devices = []

    for line in lines:
        if "NPU ID" in line:
            npu_id = line.split(":")[1].strip()
            available_devices.append(npu_id)

    if devices:
        device_list = devices.split(",")
        available_devices = [dev for dev in device_list if dev in available_devices]

    return ",".join(available_devices)

def run_train(
    runname,
    dataset,
    num_query_sample,
    num_shot,
    model_name,
    train_args,
    devices,
):
    try:
        process = subprocess.Popen(
            [
                "python",
                "train.py",
            ]
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
            env={
                **os.environ,
                "ASCEND_RT_VISIBLE_DEVICES": get_avail_devices(devices),
            },
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "out of memory" in process.stderr:
                return dataset, num_query_sample, num_shot
        else:
            return True
    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    return False


def run_eval(
    ckpt_path, dataset, num_query_sample, num_shot, model_name, npu_id, eval_args
):
    try:
        process = subprocess.Popen(
            ["python", "eval.py"]
            + merge_args(
                [
                    f"ckpt_path={ckpt_path or 'null'}",
                    f"model_name={model_name}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(eval_args),
            ),
            env={**os.environ, "ASCEND_RT_VISIBLE_DEVICES": npu_id},
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "out of memory" in process.stderr:
                return dataset, num_query_sample, num_shot
            print(
                f"Evaluation failed on npu {npu_id} for runname: {ckpt_path}, dataset: {dataset}"
            )
        else:
            return True
    except Exception as e:
        print(f"An error occurred during evaluation: {e}", file=sys.stderr)
    return False


def run_analyze(runname, dataset, num_query_sample, num_shot, model_name, analyze_args):
    if "icl" in runname:
        # runname-model-dataset
        expand_runname = f"{runname}-{model_name}-{dataset}"
    else:
        # runname-model-dataset-training_samples-num_shot
        expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
    try:
        subprocess.run(
            ["python", "analyze.py"]
            + merge_args(
                [
                    f"model_name={model_name}",
                    f"record_dir={os.path.join(paths.result_dir, 'record', expand_runname)}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(analyze_args),
            ),
            env=os.environ,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Analyze failed for runname: {runname}, dataset: {dataset}",
            file=sys.stderr,
        )
        sys.exit(1)

def execute_eval(runname, tasks, model_name, eval_args, devices):
    task_queue = tasks.copy()
    futures = {}  

    def get_npu_id():
        nonlocal futures  

        available_npus = [
            npu_id
            for npu_id in get_avail_devices(devices).split(",")
            if npu_id not in futures or futures[npu_id].done()
        ]

        if not available_npus:
            next(as_completed(futures.values())).result()
            futures = {npu_id: f for npu_id, f in futures.items() if not f.done()}
            available_npus = [
                npu_id
                for npu_id in get_avail_devices(devices).split(",")
                if npu_id not in futures
            ]

        return available_npus[0]

    with ThreadPoolExecutor() as executor:
        while task_queue or futures:
            if task_queue:
                dataset, num_query_sample, num_shot = task_queue.pop(0)
                if "icl" in runname:
                    # runname-model-dataset
                    expand_runname = f"{runname}-{model_name}-{dataset}"
                    npu_id = get_npu_id()
                    print(
                        f"Assigning task to npu {npu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, ICL"
                    )
                    futures[npu_id] = executor.submit(
                        run_eval,
                        None,
                        dataset,
                        num_query_sample,
                        num_shot,
                        model_name,
                        npu_id,
                        eval_args,
                    )
                else:
                    # runname-model-dataset-training_samples-num_shot
                    expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
                    ckpt_dir = os.path.join(paths.result_dir, "ckpt", expand_runname)
                    for epoch_ckpt in os.listdir(ckpt_dir):
                        epoch = re.findall(r"\d+", epoch_ckpt)[0]
                        npu_id = get_npu_id()
                        print(
                            f"Assigning task to npu {npu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, {epoch=}"
                        )
                        futures[npu_id] = executor.submit(
                            run_eval,
                            os.path.join(ckpt_dir, epoch_ckpt),
                            dataset,
                            num_query_sample,
                            num_shot,
                            model_name,
                            npu_id,
                            eval_args,
                        )


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation tasks.")
    parser.add_argument(
        "-r", "--runname", required=True, help="Name for the current run."
    )
    parser.add_argument(
        "-d", "--datasets", required=True, help="Comma-separated list of datasets."
    )
    parser.add_argument(
        "-m", "--model-name", required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "-q",
        "--num-query-samples",
        default="",
        help="Comma-separated list of query samples.",
    )
    parser.add_argument(
        "-s", "--num-shots", required=True, help="Comma-separated list of shots."
    )
    parser.add_argument("-t", "--train", action="store_true", help="Enable train mode.")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Enable eval mode. If set to true, analyze mode will be also enabled.",
    )
    parser.add_argument(
        "-a", "--analyze", action="store_true", help="Enable analyze mode."
    )
    parser.add_argument(
        "--train-args",
        required=False,
        help="Additional training arguments.",
    )
    parser.add_argument(
        "--eval-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--analyze-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--devices",
        required=False,
        help="Comma-separated list of integers that denotes devices used to train or evaluate.",
    )
    parser.add_argument(
        "--wait-devices-timeout",
        type=int,
        default=0,
        help="Maximum time in minutes to wait for free npus. If <= 0, exit immediately if no adequate npus are available.",
    )
    parser.add_argument(
        "--requires_memory",
        type=int,
        default=20000,
        help="The minimal npu memory used to run train or eval, unit MB.",
    )
    parser.add_argument(
        "--wait-n-devices",
        type=int,
        default=1,
        help="Minimum devices are required to start.",
    )
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    num_query_samples = args.num_query_samples.split(",")
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

    if not (args.train | args.eval | args.analyze):
        args.train = args.eval = args.analyze = True

    if (
        not get_avail_devices(devices, requires_memory)  # no even one devices
        or len(get_avail_devices(devices, requires_memory).split(","))
        < min_devices  # less than minimal required devices
    ) and (args.eval or args.train):
        print(f"Cannot find at least {min_devices} devcie(s). Start waiting...")
        for i in range(0, timeout):
            if (
                get_avail_devices(devices, requires_memory)
                and len(get_avail_devices(devices, requires_memory).split(","))
                >= min_devices
            ):
                break
            time.sleep(60)
            print(f"Waited for {min_devices} device(s) for {(i+1)} mins")
        else:
            print(f"Cannot find at least {min_devices} devcie(s). Timeout, exit...")
            return

    if num_query_samples:
        tasks = list(product(datasets, num_query_samples, num_shots))
    else:
        if args.train:
            raise RuntimeError(
                "The option -q/--num_query_samples is required if train mode is enabled."
            )
        tasks = list(product(datasets, num_shots))

    if args.train:
        print("Starting training phase...")
        task_queue = tasks.copy()
        while task_queue:
            dataset, num_query_sample, num_shot = task_queue.pop(0)
            ret = run_train(
                runname,
                dataset,
                num_query_sample,
                num_shot,
                model_name,
                train_args,
                devices,
            )

            if isinstance(ret, tuple):
                task_queue.append(ret)
            elif not ret:
                return

    if args.eval:
        print("Starting evaluation phase...")
        execute_eval(runname, tasks, model_name, eval_args, devices)

    if args.analyze:
        print("Starting analysis phase...")
        for dataset, num_query_sample, num_shot in tasks:
            run_analyze(
                runname, dataset, num_query_sample, num_shot, model_name, analyze_args
            )


if __name__ == "__main__":
    main()
