import json
import os
from typing import Any, cast

import hydra
from dotenv import load_dotenv
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.loggers import EvaluationTracker
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import get_datetime_str, make_table
from omegaconf import DictConfig, OmegaConf

import src.lmms_eval_model  # noqa: F401
from src.utils.common import resolve_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def default_model_args(cfg: DictConfig) -> tuple[str, str] | tuple[None, None]:
    model_env = {
        "idefics-9b": "IDEFICS_9B_PATH",
        "idefics2-8b-base": "IDEFICS2_8B_BASE_PATH",
        "llava-interleave-7b": "LLAVA_INTERLEAVE_7B_PATH",
    }
    if cfg.model_name not in model_env:
        return None, None

    env_key = model_env[cfg.model_name]
    if not os.environ.get(env_key):
        raise ValueError(
            f"Set {env_key} in .env or provide lmms_eval.model_args explicitly."
        )
    pretrained = resolve_path(os.environ[env_key])
    model_args = f"model_name={cfg.model_name},pretrained={pretrained}"
    if cfg.ckpt_path:
        model_args += f",peft={resolve_path(cfg.ckpt_path)}"
    return "mimic", model_args


def build_lmms_eval_config(cfg: DictConfig) -> dict[str, Any]:
    lmms_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.lmms_eval, resolve=True))
    default_model, default_args = default_model_args(cfg)
    lmms_cfg["model"] = lmms_cfg.get("model") or default_model
    lmms_cfg["model_args"] = lmms_cfg.get("model_args") or default_args

    env = {
        key: resolve_path(value) if key.endswith("_DIR") and value else value
        for key, value in lmms_cfg.pop("env", {}).items()
        if value
    }
    for key, value in env.items():
        os.environ[key] = value
    return {key: value for key, value in lmms_cfg.items() if value is not None}


def resolve_tasks(tasks: str | list[str], task_manager: TaskManager) -> list[Any]:
    if isinstance(tasks, str):
        task_list = tasks.split(",")
    else:
        task_list = tasks
    matched_tasks = task_manager.match_tasks(task_list)
    missing_tasks = [task for task in task_list if task not in matched_tasks]
    if missing_tasks:
        raise ValueError(f"Tasks not found: {', '.join(missing_tasks)}")
    return matched_tasks


@hydra.main(config_path="configs", config_name="eval.yaml", version_base=None)
def main(cfg: DictConfig):
    lmms_cfg = build_lmms_eval_config(cfg)
    output_path = resolve_path(lmms_cfg.pop("output_path"))
    verbosity = lmms_cfg.pop("verbosity", "INFO")
    task_manager = TaskManager(verbosity, model_name=lmms_cfg["model"])
    tasks = resolve_tasks(lmms_cfg.pop("tasks"), task_manager)
    datetime_str = get_datetime_str()
    tracker = EvaluationTracker(output_path=output_path)

    results = simple_evaluate(
        **lmms_cfg,
        tasks=tasks,
        task_manager=task_manager,
        evaluation_tracker=tracker,
        verbosity=verbosity,
        datetime_str=datetime_str,
    )
    if results is None:
        return

    samples = results.pop("samples", {}) if lmms_cfg.get("log_samples", True) else {}
    print(json.dumps(results, indent=4, default=str))
    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))

    tracker.save_results_aggregated(
        results=results,
        samples=samples,
        datetime_str=datetime_str,
    )
    for task_name, task_samples in samples.items():
        tracker.save_results_samples(task_name=task_name, samples=task_samples)


if __name__ == "__main__":
    load_dotenv()
    main()
