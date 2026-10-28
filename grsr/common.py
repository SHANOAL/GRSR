"""Shared dataset checks used by the GRSR entry points."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def normalize_cli_aliases(argv: Sequence[str]) -> None:
    """Accept kebab-case spellings while preserving the legacy CLI parser."""
    aliases = {
        "--data-root": "--data_root",
        "--sup-dir": "--sup_dir",
        "--save-tag": "--save_tag",
        "--output-dir": "--output_dir",
        "--coarse-threshold": "--coarse_threshold",
        "--sigma-candidates": "--sigma_candidates",
        "--lambda-contrast": "--lambda_contrast",
        "--lambda-regularization": "--lambda_regularization",
        "--window-size": "--window_size",
        "--tophat-kernel": "--tophat_kernel",
        "--r-scales": "--r_scales",
        "--sigma-search-min": "--sigma_search_min",
        "--sigma-search-max": "--sigma_search_max",
        "--sigma-reg-min": "--sigma_reg_min",
        "--sigma-reg-max": "--sigma_reg_max",
        "--support-threshold": "--support_threshold",
        "--base-size": "--base_size",
        "--crop-size": "--crop_size",
        "--batch-size": "--batch_size",
        "--num-workers": "--num_workers",
        "--deep-supervision": "--deep_supervision",
        "--focusing-gamma": "--focusing_gamma",
        "--aiou-alpha": "--aiou_alpha",
        "--gamma-high": "--gamma_high",
        "--gamma-low": "--gamma_low",
    }
    for index, value in enumerate(argv):
        if value in aliases:
            argv[index] = aliases[value]
            continue
        for alias, canonical in aliases.items():
            prefix = alias + "="
            if value.startswith(prefix):
                argv[index] = canonical + value[len(alias):]
                break


def option_value(argv: Sequence[str], names: Iterable[str]) -> Optional[str]:
    """Read an option from ``--name value`` or ``--name=value`` syntax."""
    names = tuple(names)
    for index, value in enumerate(argv):
        if value in names:
            if index + 1 >= len(argv):
                raise SystemExit(f"Option {value} requires a value")
            return argv[index + 1]
        for name in names:
            prefix = name + "="
            if value.startswith(prefix):
                return value[len(prefix):]
    return None


def require_data_root(argv: Sequence[str]) -> Path:
    """Require an explicit dataset path instead of using the stale legacy default."""
    value = option_value(argv, ("--data_root", "--data-root"))
    if not value:
        raise SystemExit(
            "Please pass --data-root, e.g. --data-root data/SIRST-v1"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")
    return root


def resolve_layout(data_root: Path) -> Tuple[Path, Path]:
    """Resolve either IRSTD-1k or generic images/masks directory layout."""
    candidates = (
        (data_root / "IRSTD1k_Img", data_root / "IRSTD1k_Label"),
        (data_root / "images", data_root / "masks"),
    )
    for image_dir, mask_dir in candidates:
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir
    expected = "IRSTD1k_Img + IRSTD1k_Label, or images + masks"
    raise SystemExit(f"Unrecognized dataset layout: {data_root}\nExpected: {expected}")


def resolve_split_path(data_root: Path, name: str) -> Path:
    """Resolve a train/test split in either the flat or the BasicIRSTD layout.

    The BasicIRSTD bundle keeps split files under ``img_idx`` and names them
    ``train_<dataset>.txt``/``test_<dataset>.txt``.  GRSR also accepts the
    simpler root-level ``trainval.txt``/``test.txt`` names, so data preparation
    may provide symlinks but the runner should also work directly on an
    official extraction.  Ambiguous files are rejected instead of silently
    selecting a different protocol.
    """
    root = Path(data_root)
    direct = root / name
    if direct.is_file():
        return direct

    split = "train" if name == "trainval.txt" else "test" if name == "test.txt" else None
    if split is None:
        raise SystemExit(f"Unsupported split name: {name}")

    index_dir = root / "img_idx"
    candidates = [index_dir / f"{split}.txt"]
    candidates.extend(sorted(index_dir.glob(f"{split}_*.txt")))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise SystemExit(
            f"Missing split file: {direct} (and no {index_dir}/{split}_*.txt)"
        )
    if len(candidates) == 1:
        return candidates[0]

    # Prefer a file whose stem identifies this dataset.  SIRST-v1 is the
    # BasicIRSTD name for NUAA-SIRST, hence the explicit alias below.
    aliases = {root.name.lower(), root.name.lower().replace("-", "")}
    if root.name.lower() in {"sirst-v1", "nuaa-sirst"}:
        aliases.update({"nuaa-sirst", "nuaa", "sirst"})
    elif root.name.lower() == "nudt-sirst":
        aliases.update({"nudt-sirst", "nudt"})
    elif root.name.lower() == "irstd-1k":
        aliases.update({"irstd-1k", "irstd1k", "irstd"})

    matched = [
        path
        for path in candidates
        if any(alias in path.stem.lower().replace("_", "-") for alias in aliases)
    ]
    if len(matched) == 1:
        return matched[0]
    listed = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"Ambiguous split files, please provide {direct}: {listed}")


def require_split(data_root: Path, name: str) -> Path:
    return resolve_split_path(data_root, name)


def count_ids(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def print_dataset_summary(data_root: Path, need_splits: bool) -> None:
    image_dir, mask_dir = resolve_layout(data_root)
    image_count = sum(1 for path in image_dir.iterdir() if path.suffix.lower() == ".png")
    mask_count = sum(1 for path in mask_dir.iterdir() if path.suffix.lower() == ".png")
    print(f"[data] root={data_root}")
    print(f"[data] images={image_dir} ({image_count} PNG)")
    print(f"[data] masks={mask_dir} ({mask_count} PNG)")
    if image_count == 0 or mask_count == 0:
        raise SystemExit("Image or mask directory is empty.")
    if image_count != mask_count:
        print(f"[warning] image/mask count mismatch: {image_count} vs {mask_count}")

    if need_splits:
        train_file = resolve_split_path(data_root, "trainval.txt")
        test_file = resolve_split_path(data_root, "test.txt")
        print(f"[split] trainval={count_ids(train_file)}, test={count_ids(test_file)}")


def resolve_supervision_dir(data_root: Path, argv: Sequence[str]) -> Path:
    custom = option_value(argv, ("--sup_dir", "--sup-dir"))
    if custom:
        path = Path(custom).expanduser().resolve()
    else:
        path = data_root / "supervision_maps_grsr" / "initial"
    if not path.is_dir():
        raise SystemExit(
            f"Pseudo-label directory not found: {path}\n"
            "Run generate_pseudo_labels.py first, or pass --sup-dir."
        )
    pseudo_count = sum(1 for item in path.iterdir() if item.suffix.lower() == ".png")
    if pseudo_count == 0:
        raise SystemExit(f"Pseudo-label directory is empty: {path}")
    print(f"[pseudo] dir={path} ({pseudo_count} PNG)")
    return path
