"""FOMO300K brain-age dataset loader for FlowLet training."""

import gzip
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from flowlet.data.openbhb_dataset import OpenBHBDataset


class FOMO300KDataset(OpenBHBDataset):
    """Load T1w NIfTI scans packaged by session in the FOMO300K repository.

    The local dataset is organized as ZIP files rather than as a ``datasets``
    table. ``participants.tsv`` supplies ages and ``mapping.tsv`` links T1w
    acquisitions to session ZIP files under ``data_dir``.
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

        if metadata.empty:
            raise ValueError("No T1w FOMO300K scans with age metadata were found.")

        return metadata.reset_index(drop=True)

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

    def _load_volume(self, row: pd.Series) -> np.ndarray:
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
            compressed_nifti = archive.read(candidates[0])

        nifti_bytes = gzip.decompress(compressed_nifti)
        volume = nib.Nifti1Image.from_bytes(nifti_bytes).get_fdata(dtype=np.float32)
        volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D T1w volume, got shape {volume.shape}.")
        return volume
