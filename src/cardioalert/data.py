"""Download the public UCI archive and preserve auditable file/row provenance."""

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

from .schema import FEATURES

SOURCE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
SOURCE_DOI = "https://doi.org/10.24432/C52P4X"
SITES = ("cleveland", "hungarian", "switzerland", "va")
FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_data(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "CardioAlert-research/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(5_000_001)
    if len(raw) > 5_000_000:
        raise ValueError("Unexpectedly large UCI archive")
    manifest = {
        "source_url": SOURCE_URL, "doi": SOURCE_DOI,
        "license": "CC BY 4.0", "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "files": {},
    }
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for site, filename in FILES.items():
            # Read only explicitly named files; never extract arbitrary archive paths.
            content = archive.read(filename)
            path = directory / filename
            path.write_bytes(content)
            manifest["files"][site] = {"filename": filename, "sha256": sha256_file(path)}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_data(directory: Path, sites: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    frames = []
    hashes = {}
    for site in sites:
        path = directory / FILES[site]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run: cardioalert download-data")
        frame = pd.read_csv(path, header=None, na_values="?")
        if frame.shape[1] != 14:
            raise ValueError(f"{path}: expected exactly 14 columns")
        frame.columns = FEATURES + ["diagnosis"]
        frame = frame.apply(pd.to_numeric, errors="raise")
        if frame["diagnosis"].isna().any() or not frame["diagnosis"].isin([0, 1, 2, 3, 4]).all():
            raise ValueError(f"{path}: missing or invalid target label")
        frame["target"] = (frame.pop("diagnosis") > 0).astype(int)
        frame["site"] = site
        frame["record_id"] = [f"{site}:{i + 1}" for i in range(len(frame))]
        frames.append(frame)
        hashes[site] = sha256_file(path)
    frame = pd.concat(frames, ignore_index=True)
    # Some processed sites encode unavailable measurements as zero. Treat those
    # as missing, not as a measured blood pressure/cholesterol/heart rate of zero.
    invalid_measurements = {}
    for name in ("trestbps", "chol", "thalach"):
        mask = frame[name] <= 0
        invalid_measurements[name] = int(mask.sum())
        frame.loc[mask, name] = float("nan")
    raw_count = len(frame)
    # Keep one occurrence of each feature vector, even if a duplicated vector has
    # conflicting labels. The number removed is reported, never silently resampled.
    frame = frame.drop_duplicates(subset=FEATURES, keep="first").reset_index(drop=True)
    metadata = {
        "source_url": SOURCE_URL, "doi": SOURCE_DOI, "license": "CC BY 4.0",
        "sites": list(sites), "raw_rows": raw_count, "rows": len(frame),
        "duplicate_feature_rows_removed": raw_count - len(frame), "file_sha256": hashes,
        "site_counts": frame["site"].value_counts().sort_index().to_dict(),
        "missing_by_feature": frame[FEATURES].isna().sum().to_dict(),
        "nonpositive_measurements_set_missing": invalid_measurements,
    }
    return frame, metadata
