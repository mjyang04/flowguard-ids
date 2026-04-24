"""CICIDS2017 CSV loader.

Reads the eight raw day-split CSVs distributed by the Canadian Institute
for Cybersecurity (Sharafaldin et al., 2018), handles the well-documented
schema quirks, and returns the same :class:`DataSplits` shape as
:func:`nids.data.lycos.get_data`.

Intentional design choices for the label-quality comparison study:

* **Labels are preserved as-is.** We do NOT apply the timing-based relabel
  from Engelen et al. (2021) or the LycoSTand feature-extraction fix from
  Rosay et al. (2022). The whole point of running CLAN on this dataset is
  to measure the effect of those errors; "fixing" them here would defeat
  the control.
* **Column whitespace is normalised only enough to make the CSV parseable**
  (strip leading/trailing whitespace from column names). The well-known
  leading-space ``" Label"`` quirk is resolved by this single ``.strip()``.
* **Unicode en-dash in label values** (``Web Attack – Brute Force``) is
  normalised by :func:`_coerce_label` downstream.

Known CICIDS2017 issues inherited into this loader — documented in the
thesis Chapter 2 / Chapter 4.5 discussion, NOT silently patched:

1. Duplicate flows (~305k flows are exact duplicates — Engelen 2021).
2. Label leakage across time windows (benign flows inside attack windows
   are mis-labelled as the attack class — Engelen 2021, Rosay 2023).
3. CICFlowMeter v3 extraction bugs (negative durations, oversize packet
   counts — Rosay 2023).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ._core import DataSplits, prepare_splits

logger = logging.getLogger(__name__)

__all__ = ["get_data_cicids", "load_cicids_dataframe"]


CICIDS_META_COLS: tuple[str, ...] = (
    "flow_id",
    "source_ip",
    "source_port",
    "destination_ip",
    "destination_port",
    "protocol",
    "timestamp",
)

# Map raw upper-case CICIDS2017 labels to the lower-case snake-case naming
# used by Lycos2017, so the comparison is on the same label space.
_LABEL_CANONICAL: dict[str, str] = {
    "benign": "benign",
    "ddos": "ddos",
    "dos goldeneye": "dos_goldeneye",
    "dos hulk": "dos_hulk",
    "dos slowhttptest": "dos_slowhttptest",
    "dos slowloris": "dos_slowloris",
    "ftp-patator": "ftp_patator",
    "ssh-patator": "ssh_patator",
    "portscan": "portscan",
    "bot": "botnet",
    "heartbleed": "heartbleed",
    "infiltration": "infiltration",
    "web attack - brute force": "web_attack_brute_force",
    "web attack - sql injection": "web_attack_sql_injection",
    "web attack - xss": "web_attack_xss",
}


def _normalise_column_name(name: str) -> str:
    """Strip whitespace, lower-case, and convert to snake_case."""
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("/", "_per_").replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _canonicalise_label(raw: object) -> str:
    """Map a raw CICIDS2017 label value to a Lycos2017-aligned canonical form.

    Handles the ``Web Attack – Brute Force`` en-dash variants and the
    mojibake ``Web Attack \\xef\\xbf\\xbd Brute Force`` form produced when
    the CSV is opened with the wrong encoding by collapsing any non-ASCII
    run to a single ``-``.
    """
    if pd.isna(raw):
        return "unknown"
    text = str(raw).strip()
    # Collapse every non-ASCII character (en-dash, replacement char, etc.)
    # into a single ASCII hyphen so the lookup table hits.
    ascii_text = "".join(ch if ord(ch) < 128 else "-" for ch in text)
    key = ascii_text.lower().strip()
    # Normalise repeated whitespace introduced by the collapse.
    while "  " in key:
        key = key.replace("  ", " ")
    # Unified whitespace around the hyphen used by Web Attack classes.
    key = key.replace(" - ", " - ").replace("- ", "- ").replace(" -", " -")
    return _LABEL_CANONICAL.get(key, key.replace(" ", "_").replace("-", "_"))


def _read_csv_autoenc(handle) -> pd.DataFrame:
    """Try utf-8 first, fall back to latin-1 — real CICIDS2017 is latin-1
    but tests write utf-8 fixtures with an en-dash that latin-1 cannot
    encode. Reading is symmetric: try the lossless encoding first."""
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(handle, "seek"):
                handle.seek(0)
            return pd.read_csv(handle, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    # Give up gracefully: re-raise the last attempt's error.
    return pd.read_csv(handle, encoding="latin-1", low_memory=False)


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single CICIDS2017 CSV, whether zipped or extracted."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV member inside {path}")
            with zf.open(members[0]) as f:
                df = _read_csv_autoenc(f)
    else:
        df = _read_csv_autoenc(path)
    df.columns = [_normalise_column_name(c) for c in df.columns]
    return df


def load_cicids_dataframe(
    source: str | Path,
    *,
    glob_pattern: str = "*.pcap_ISCX.zip",
) -> pd.DataFrame:
    """Read and concatenate every CICIDS2017 day CSV/zip under ``source``.

    ``source`` may be (a) a single CSV, (b) a single zip, or (c) a directory
    containing the eight day-split archives. Column names are normalised in
    every file before concatenation so that the mixed whitespace casing
    across the CIC archives does not cause ``NaN`` fills on concat.
    """
    path = Path(source)
    if path.is_file():
        frames = [_read_one(path)]
    elif path.is_dir():
        files = sorted(path.glob(glob_pattern))
        if not files:
            files = sorted(path.glob("*.csv")) + sorted(path.glob("*.zip"))
        if not files:
            raise FileNotFoundError(
                f"No CICIDS2017 CSV or zip files found under {path}"
            )
        logger.info("loading %d CICIDS2017 files from %s", len(files), path)
        frames = [_read_one(f) for f in files]
    else:
        raise FileNotFoundError(f"CICIDS2017 source not found: {path}")

    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Rename the label column once, after column normalisation produced
    # "label" from both " Label" and "Label".
    if "label" not in df.columns:
        raise KeyError(
            f"Expected a 'label' column after normalisation; have {df.columns.tolist()}"
        )

    # CICIDS2017 has a well-documented defect: the Thursday-Morning-WebAttacks
    # archive contains ~288k rows with an empty label field (Engelen et al.,
    # 2021). Dropping these is standard practice and does not compromise the
    # noisy-label control argument — these are *missing* labels rather than
    # *wrong* labels, and no downstream method can consume them.
    n_before = len(df)
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("dropped %d rows with empty label (CICIDS2017 defect)", n_dropped)

    df["label"] = df["label"].map(_canonicalise_label)

    # Replace Inf with NaN so nan_to_num in prepare_splits can handle it.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_data_cicids(
    data_source: str | Path,
    *,
    target: str = "label",
    drop: Iterable[str] = CICIDS_META_COLS,
    class_zero: str = "benign",
    sample_thres: int = 100,
    split_seed: int = 39058032,
    test_ratio: float = 0.5,
    val_ratio: float = 0.0,
    anomaly_detection: bool = True,
    standardise: bool = True,
) -> DataSplits:
    """Load CICIDS2017 and produce the 8 splits consumed by CLAN scripts.

    ``data_source`` may be a directory of day-split zips/CSVs or a single
    pre-merged CSV. Labels are kept in their original (potentially noisy)
    form — this is the control for the Lycos2017 comparison.
    """
    df = load_cicids_dataframe(data_source)
    return prepare_splits(
        df,
        target=target,
        drop=drop,
        class_zero=class_zero,
        sample_thres=sample_thres,
        split_seed=split_seed,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        anomaly_detection=anomaly_detection,
        standardise=standardise,
        drop_zero_cols=True,
    )
