"""Model trainer and exporter wrapper with large-sample auto-selection and metrics tracking."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from plateai_trainer.training.control import TrainingProgress
from .paths import package_data_root


_PACKAGED_DATA_ROOT = package_data_root()
_TASK_ID_RE = re.compile(r"[0-9a-f]{12}\Z")
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_DEFAULT_CHARSET = "configs/charsets/tw_new_style_private_passenger_v1.txt"
_DEFAULT_RULES = "configs/plate_rules/tw_new_style_private_passenger_v1.json"
_DEFAULT_TEMPLATE = "configs/plate_templates/new_style_private_passenger_white_v1.json"
_DEFAULT_AUGMENTATION = "configs/augmentation/standard_v1.json"


def _resolve_within(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must stay within the project root") from exc
    return resolved


def _resolve_config_path(root: Path, relative_path: str, *, field: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{field} must reference a bundled configuration file")
    for base in (root, _PACKAGED_DATA_ROOT):
        candidate = (base / path).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    raise ValueError(f"{field} does not resolve to a configuration file: {relative_path}")


def _dataset_contract(root: Path, dataset: Path) -> dict[str, object]:
    config_path = dataset / "generation_config.json"
    if not config_path.is_file():
        raise ValueError(f"dataset is missing generation_config.json: {dataset}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"dataset generation_config.json is unreadable: {dataset}") from exc
    if not isinstance(document, dict) or type(document.get("seed")) is not int:
        raise ValueError(f"dataset generation_config.json has no integer seed: {dataset}")
    config_paths = document.get("config_paths")
    if not isinstance(config_paths, dict):
        raise ValueError(f"dataset generation_config.json has no config_paths: {dataset}")
    charset = config_paths.get("charset")
    rules = config_paths.get("rules")
    if not isinstance(charset, str) or not isinstance(rules, str):
        raise ValueError(f"dataset generation_config.json lacks charset/rules paths: {dataset}")
    return {
        "path": dataset,
        "seed": document["seed"],
        "charset": _resolve_config_path(root, charset, field="dataset charset"),
        "rules": _resolve_config_path(root, rules, field="dataset rules"),
    }


def _compatible_validation(
    train: Mapping[str, object], validation: Mapping[str, object]
) -> bool:
    return (
        validation["path"] != train["path"]
        and validation["seed"] != train["seed"]
        and validation["charset"] == train["charset"]
        and validation["rules"] == train["rules"]
    )


def _validate_run_name(value: object, task_id: str) -> str:
    run_name = f"run-{task_id}" if value is None else value
    if not isinstance(run_name, str) or not _RUN_NAME_RE.fullmatch(run_name):
        raise ValueError("run_name must use only letters, digits, '.', '_' or '-' and no paths")
    if run_name.endswith(".") or run_name.lower() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("run_name is not a portable Windows directory name")
    return run_name


def _positive_integer(value: object, *, field: str, default: int, maximum: int) -> int:
    parsed = default if value is None else value
    if type(parsed) is not int or not 1 <= parsed <= maximum:
        raise ValueError(f"{field} must be an integer from 1 to {maximum}")
    return parsed


def _validate_device(value: object) -> str:
    device = "auto" if value is None else value
    if not isinstance(device, str) or not re.fullmatch(r"(?:auto|cpu|cuda|cuda:[0-9]+)", device):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    return device


def validate_training_request(root: Path, task_id: str, request: dict) -> dict:
    """Validate a Web training request without creating or replacing artifacts."""

    if not isinstance(request, dict):
        raise ValueError("training request must be an object")
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("task_id must be a 12-character lowercase hexadecimal id")

    root = root.resolve()
    run_name = _validate_run_name(request.get("run_name"), task_id)
    run_directory = root / "runs" / run_name
    bundle_directory = root / "models" / "bundles" / f"train-{task_id}"
    if run_directory.exists():
        raise FileExistsError(f"training output already exists: {run_directory}")
    if bundle_directory.exists():
        raise FileExistsError(f"model bundle output already exists: {bundle_directory}")

    train_value = request.get("train_dataset")
    generated_train = train_value is None
    train_directory = root / "out" / "train-default" if generated_train else _resolve_within(
        root, train_value, field="train_dataset"
    )
    if not generated_train and not train_directory.is_dir():
        raise ValueError(f"train_dataset does not exist: {train_directory}")
    if generated_train and train_directory.exists() and not train_directory.is_dir():
        raise ValueError(f"default train dataset is not a directory: {train_directory}")

    train_contract: dict[str, object]
    if generated_train and not train_directory.exists():
        train_contract = {
            "path": train_directory,
            "seed": 42,
            "charset": _resolve_config_path(root, _DEFAULT_CHARSET, field="default charset"),
            "rules": _resolve_config_path(root, _DEFAULT_RULES, field="default rules"),
        }
    else:
        train_contract = _dataset_contract(root, train_directory)

    validation_value = request.get("validation_dataset")
    generated_validation = False
    if validation_value is not None:
        validation_directory = _resolve_within(root, validation_value, field="validation_dataset")
        if not validation_directory.is_dir():
            raise ValueError(f"validation_dataset does not exist: {validation_directory}")
        validation_contract = _dataset_contract(root, validation_directory)
        if not _compatible_validation(train_contract, validation_contract):
            raise ValueError("validation_dataset must use a different seed and matching charset/rules")
    else:
        validation_contract = None
        out_directory = root / "out"
        if out_directory.is_dir():
            for candidate in sorted(out_directory.iterdir(), key=lambda path: path.name.lower()):
                if not candidate.is_dir() or "val" not in candidate.name.lower():
                    continue
                try:
                    candidate_contract = _dataset_contract(root, candidate.resolve())
                except ValueError:
                    continue
                if _compatible_validation(train_contract, candidate_contract):
                    validation_directory = candidate.resolve()
                    validation_contract = candidate_contract
                    break
        if validation_contract is None:
            generated_validation = True
            validation_directory = root / "out" / f"val-{task_id}"
            if validation_directory.exists():
                raise FileExistsError(
                    f"generated validation output already exists: {validation_directory}"
                )

    return {
        "task_id": task_id,
        "epochs": _positive_integer(request.get("epochs"), field="epochs", default=5, maximum=100),
        "batch_size": _positive_integer(
            request.get("batch_size"), field="batch_size", default=32, maximum=256
        ),
        "device": _validate_device(request.get("device")),
        "run_name": run_name,
        "run_directory": run_directory,
        "bundle_directory": bundle_directory,
        "train_directory": train_directory,
        "validation_directory": validation_directory,
        "generated_train": generated_train and not train_directory.exists(),
        "generated_validation": generated_validation,
        "charset_path": train_contract["charset"],
        "rules_path": train_contract["rules"],
        "train_seed": train_contract["seed"],
    }


def _generate_default_dataset(
    root: Path,
    *,
    output: Path,
    count: int,
    seed: int,
    on_progress: Callable[[TrainingProgress], None],
    check_cancelled: Callable[[], None],
    phase: str,
) -> None:
    from plateai_trainer.synthetic.dataset import generate_dataset
    from plateai_trainer.synthetic.models import GenerationRequest

    def report_progress(completed: int, total: int, _display: str) -> None:
        check_cancelled()
        on_progress(
            TrainingProgress(
                kind="batch",
                phase=phase,
                batch=completed,
                total_batches=total,
            )
        )

    check_cancelled()
    on_progress(TrainingProgress(kind="stage", phase=phase))
    generate_dataset(
        GenerationRequest(
            count=count,
            seed=seed,
            output=output,
            charset_path=_resolve_config_path(root, _DEFAULT_CHARSET, field="default charset"),
            rules_path=_resolve_config_path(root, _DEFAULT_RULES, field="default rules"),
            template_path=_resolve_config_path(root, _DEFAULT_TEMPLATE, field="default template"),
            augmentation_path=_resolve_config_path(
                root, _DEFAULT_AUGMENTATION, field="default augmentation"
            ),
        ),
        progress_callback=report_progress,
    )
    check_cancelled()


def _web_history(history: object) -> list[dict[str, object]]:
    if not isinstance(history, list):
        raise ValueError("training report history is invalid")
    result: list[dict[str, object]] = []
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("training report history entry is invalid")
        result.append(
            {
                "epoch": record["epoch"],
                "train_loss": record["train_loss"],
                "val_loss": record["val_loss"],
                "val_acc": round(float(record["val_acc"]) * 100, 1),
            }
        )
    return result


def run_training_pipeline(
    root: Path,
    task_id: str,
    request: dict,
    *,
    on_progress: Callable[[TrainingProgress], None],
    on_result: Callable[[dict], None],
    check_cancelled: Callable[[], None],
) -> dict:
    """Generate missing inputs, train, export a task-local bundle, and report real metrics."""

    from plateai_trainer.export.bundle import ExportRequest, export_crop_bundle
    from plateai_trainer.training.engine import TrainingConfig, train_recognizer

    validated = validate_training_request(root, task_id, request)
    root = root.resolve()
    if validated["generated_train"]:
        _generate_default_dataset(
            root,
            output=validated["train_directory"],
            count=2000,
            seed=int(validated["train_seed"]),
            on_progress=on_progress,
            check_cancelled=check_cancelled,
            phase="generating_train",
        )
    if validated["generated_validation"]:
        validation_seed = 999 if validated["train_seed"] != 999 else 8888
        _generate_default_dataset(
            root,
            output=validated["validation_directory"],
            count=500,
            seed=validation_seed,
            on_progress=on_progress,
            check_cancelled=check_cancelled,
            phase="generating_validation",
        )

    check_cancelled()
    training = train_recognizer(
        TrainingConfig(
            train_directory=validated["train_directory"],
            validation_directory=validated["validation_directory"],
            output_directory=validated["run_directory"],
            epochs=validated["epochs"],
            batch_size=validated["batch_size"],
            device=validated["device"],
            charset_path=validated["charset_path"],
            rules_path=validated["rules_path"],
        ),
        on_progress=on_progress,
        check_cancelled=check_cancelled,
    )
    check_cancelled()
    on_progress(TrainingProgress(kind="stage", phase="exporting"))
    bundle_directory = export_crop_bundle(
        ExportRequest(
            checkpoint=training.best_checkpoint,
            report=training.report_path,
            output=validated["bundle_directory"],
            charset_path=validated["charset_path"],
            rules_path=validated["rules_path"],
        )
    )
    check_cancelled()

    history = _web_history(training.report.get("history"))
    current_metrics = history[-1] if history else None
    result = {
        "status": "success",
        "run_dir": str(training.output_directory),
        "checkpoint": str(training.best_checkpoint),
        "report": str(training.report_path),
        "bundle_dir": str(bundle_directory),
        "history": history,
        "current_metrics": current_metrics,
        "total_epochs": validated["epochs"],
        "final_accuracy": current_metrics["val_acc"] if current_metrics else None,
        "final_loss": current_metrics["train_loss"] if current_metrics else None,
    }
    on_result(result)
    return result


def find_available_datasets(root: Path) -> list[dict[str, Any]]:
    """Scan out/ directory for available dataset candidates."""
    out_dir = root / "out"
    if not out_dir.exists():
        return []
    candidates = []
    for p in out_dir.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            images_dir = p / "images"
            count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else len(list(p.glob("*.png")))
            if count > 0:
                candidates.append({
                    "name": p.name,
                    "path": str(p),
                    "count": count,
                })
    candidates.sort(key=lambda x: (x["name"] == "demo-10000", x["count"]), reverse=True)
    return candidates


def get_dataset_seed(path: Path) -> int | None:
    """Return a generated dataset's seed when its metadata is valid JSON."""

    try:
        with (Path(path) / "generation_config.json").open("r", encoding="utf-8") as stream:
            seed = json.load(stream).get("seed")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return seed if type(seed) is int else None


def get_dataset_config_paths(path: Path) -> dict[str, str]:
    """Return the recorded portable config paths without mutating a dataset."""

    try:
        with (Path(path) / "generation_config.json").open("r", encoding="utf-8") as stream:
            config_paths = json.load(stream).get("config_paths")
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return dict(config_paths) if isinstance(config_paths, dict) else {}
