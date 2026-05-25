"""FOMO300K brain-age dataset loader for FlowLet training."""

import gzip
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from flowlet.data.openbhb_dataset import OpenBHBDataset


class FOMO300KDataset(OpenBHBDataset):
    """Load age-labelled local T1w scans packaged as ZIP files.

    ``participants.tsv`` supplies ages and ``mapping.tsv`` links T1w
    acquisitions to session ZIP files under ``data_dir``. An optional
    preflight validates each selected NIfTI member and caches validation
    results using the ZIP size and modification timestamp.
    """

    def __init__(
        self,
        data_dir: str = "/root/DATASETS/FOMO300K_brain_age",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        age_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        augment: bool = False,
        cache_data: bool = False,
        include_site: bool = False,
        max_samples: Optional[int] = None,
        validate_archives: bool = True,
        validation_cache_file: Optional[str] = None,
        validation_workers: int = 4,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"FOMO300K data directory not found: {self.data_dir}")
        self.target_shape = target_shape
        self.normalize = normalize
        self.augment = augment
        self.cache_data = cache_data
        self.include_site = include_site
        self.max_samples = max_samples
        self.validate_archives = validate_archives
        self.validation_cache_file = Path(validation_cache_file) if validation_cache_file else (
            self.data_dir / ".flowlet_t1w_validation.csv"
        )
        self.validation_workers = validation_workers
        self.transform = transform
        self.filter_diagnosis = None
        self._monai_transform = None

        self.metadata = self._load_fomo_metadata()
        if age_range is None:
            ages = self.metadata["age"].values
            self.age_range = (float(ages.min()), float(ages.max()))
        else:
            self.age_range = age_range

        self._cache: Optional[Dict[int, Dict]] = {} if cache_data else None

        print(f"Loaded FOMO300K dataset with {len(self)} T1w scans")
        print(f"Age range: {self.age_range}")
        if include_site:
            print(f"Number of source datasets: {self.metadata['site'].nunique()}")

    def _load_fomo_metadata(self) -> pd.DataFrame:
        participants = pd.read_csv(self.data_dir / "participants.tsv", sep="\t")
        participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
        participants = participants.dropna(subset=["age"])
        print(f"Selected {len(participants)} sessions with numeric age metadata")

        mapping = pd.read_csv(self.data_dir / "mapping.tsv", sep="\t")
        t1w = mapping[
            mapping["new_filename"].astype(str).str.endswith("_T1w.nii.gz", na=False)
        ].copy()
        metadata = t1w.merge(
            participants[["dataset", "participant_id", "session_id", "age", "group"]],
            on=["dataset", "participant_id", "session_id"],
            how="inner",
        )
        metadata = metadata.dropna(subset=["new_filename"])
        metadata["site"] = metadata["dataset"]
        metadata["zip_path"] = (
            metadata["dataset"]
            + "/"
            + metadata["participant_id"]
            + "/"
            + metadata["session_id"]
            + ".zip"
        )

        before_filter = len(metadata)
        metadata = metadata[metadata["zip_path"].map(lambda path: (self.data_dir / path).is_file())]
        removed = before_filter - len(metadata)
        if removed:
            print(
                f"Filtered {removed} T1w scans whose ZIP files are not present "
                f"under {self.data_dir}"
            )

        if self.max_samples is not None:
            metadata = metadata.iloc[: self.max_samples]

        if self.validate_archives:
            metadata = self._filter_valid_archives(metadata)

        if metadata.empty:
            raise ValueError("No valid T1w FOMO300K scans with age metadata were found.")

        return metadata.reset_index(drop=True)

    def _filter_valid_archives(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata = metadata.copy()
        stats = metadata["zip_path"].map(lambda path: (self.data_dir / path).stat())
        metadata["zip_size"] = stats.map(lambda stat: stat.st_size)
        metadata["zip_mtime_ns"] = stats.map(lambda stat: stat.st_mtime_ns)
        keys = ["zip_path", "new_filename", "zip_size", "zip_mtime_ns"]

        cached = pd.DataFrame(columns=keys + ["is_valid", "error"])
        if self.validation_cache_file.exists():
            cached = pd.read_csv(self.validation_cache_file)
            cached = cached.drop_duplicates(subset=keys, keep="last")

        cached_keys = set(map(tuple, cached[keys].itertuples(index=False, name=None)))
        checks = metadata.drop_duplicates(subset=keys)
        pending = [row for _, row in checks.iterrows() if tuple(row[key] for key in keys) not in cached_keys]

        if pending:
            print(f"Validating {len(pending)} uncached FOMO300K T1w archive members before training...")
            if self.validation_workers > 1:
                with ThreadPoolExecutor(max_workers=self.validation_workers) as executor:
                    results = executor.map(self._validate_archive_row, pending)
                    new_records = self._collect_validation_results(results, len(pending))
            else:
                results = map(self._validate_archive_row, pending)
                new_records = self._collect_validation_results(results, len(pending))
            cached = pd.concat([cached, pd.DataFrame(new_records)], ignore_index=True)
            cached = cached.drop_duplicates(subset=keys, keep="last")
            self.validation_cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.validation_cache_file.with_suffix(self.validation_cache_file.suffix + ".tmp")
            cached.to_csv(tmp_file, index=False)
            tmp_file.replace(self.validation_cache_file)
        else:
            print(f"Using cached archive validation from {self.validation_cache_file}")

        validity = {
            tuple(row[key] for key in keys): str(row["is_valid"]).lower() == "true"
            for _, row in cached.iterrows()
        }
        valid_mask = metadata.apply(lambda row: validity.get(tuple(row[key] for key in keys), False), axis=1)
        invalid = metadata[~valid_mask]
        if not invalid.empty:
            print(f"Filtered {len(invalid)} T1w scans with corrupt or unreadable local archive members")
        return metadata[valid_mask].drop(columns=["zip_size", "zip_mtime_ns"])

    @staticmethod
    def _collect_validation_results(results, total: int):
        records = []
        for count, result in enumerate(results, start=1):
            records.append(result)
            if count % 100 == 0 or count == total:
                print(f"Validated {count}/{total} archive members")
        return records

    def _validate_archive_row(self, row: pd.Series) -> Dict:
        record = {
            "zip_path": row["zip_path"],
            "new_filename": row["new_filename"],
            "zip_size": row["zip_size"],
            "zip_mtime_ns": row["zip_mtime_ns"],
            "is_valid": True,
            "error": "",
        }
        try:
            gzip.decompress(self._read_compressed_nifti(row))
        except (gzip.BadGzipFile, zipfile.BadZipFile, EOFError, ValueError, zlib.error) as exc:
            record["is_valid"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"Invalid FOMO300K archive member {row['zip_path']}: {record['error']}")
        return record

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None and idx in self._cache:
            data = self._cache[idx].copy()
            if self.augment:
                data["volume"] = self._apply_augmentation(data["volume"])
            return data

        row = self.metadata.iloc[idx]
        volume = self._preprocess(self._load_volume(row))
        data = {
            "volume": torch.from_numpy(volume).float().unsqueeze(0),
            "age": torch.tensor([row["age"]]).float(),
            "participant_id": f"{row['dataset']}/{row['participant_id']}/{row['session_id']}",
        }
        if self.include_site:
            data["site"] = row["site"]
        if self.transform is not None:
            data = self.transform(data)

        if self._cache is not None:
            self._cache[idx] = {
                "volume": data["volume"].clone(),
                "age": data["age"].clone(),
                "participant_id": data["participant_id"],
            }
            if self.include_site:
                self._cache[idx]["site"] = data["site"]

        if self.augment:
            data["volume"] = self._apply_augmentation(data["volume"])
        return data

    def _read_compressed_nifti(self, row: pd.Series) -> bytes:
        zip_path = self.data_dir / row["zip_path"]
        with zipfile.ZipFile(zip_path) as archive:
            candidates = [
                member for member in archive.namelist() if member.endswith("/" + row["new_filename"])
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one '{row['new_filename']}' in {row['zip_path']}, "
                    f"found {len(candidates)}."
                )
            return archive.read(candidates[0])

    def _load_volume(self, row: pd.Series) -> np.ndarray:
        nifti_bytes = gzip.decompress(self._read_compressed_nifti(row))
        volume = nib.Nifti1Image.from_bytes(nifti_bytes).get_fdata(dtype=np.float32)
        volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D T1w volume, got shape {volume.shape}.")
        return volume
