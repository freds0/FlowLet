# Training With FOMO300K Brain Age

The `flowlet_fomo300k` experiment reads the dataset already downloaded at
`/root/DATASETS/FOMO300K_brain_age`. It selects sessions with numeric age
metadata before indexing T1w NIfTI acquisitions, and only reads ZIP files
that are present locally.

## Install

```bash
pip install -r requirements-fomo300k.txt
pip install -e .
```

## Train

```bash
python flowlet/train.py experiment=flowlet_fomo300k
```

The T1w subset contains 29,118 acquisitions with recorded ages from -17 to 96
years, including prenatal scans. The experiment config sets the model's age
normalization to that full range instead of clipping it to the OpenBHB default.

For a local smoke test before starting a full run:

```bash
python flowlet/train.py experiment=flowlet_fomo300k \
  data.max_samples=32 data.num_workers=0 data.augment_train=false \
  trainer=cpu trainer.fast_dev_run=true logger=csv
```

Override `data.data_dir=/path/to/FOMO300K_brain_age` when the local dataset is stored elsewhere.

## Corrupt ZIP Files

Training skips a ZIP that raises a CRC or gzip integrity error and uses another
sample from the same split. Each skipped path is emitted as a warning. This is
enabled by default through `data.skip_corrupt_files=true`; set it to `false`
to stop immediately when auditing the local copy.

A reported file should still be replaced in the local dataset. To verify one
specific ZIP before restarting training:

```bash
python - <<'PY'
import zipfile
path = "/root/DATASETS/FOMO300K_brain_age/PATH/TO/ses-01.zip"
with zipfile.ZipFile(path) as archive:
    print(archive.testzip() or "ZIP OK")
PY
```

## Preflight Validation

Run the integrity scan before starting a full training job:

```bash
python scripts/validate_fomo300k.py \
  --data-dir /root/DATASETS/FOMO300K_brain_age \
  --workers 4
```

The scan reads each selected T1w member fully to validate ZIP and gzip CRCs.
Its first execution can take substantial time because it audits the local data.
Results are stored in `.flowlet_t1w_validation.csv` inside the dataset directory;
subsequent training runs reuse entries while the ZIP size and modification time
remain unchanged. Validation is also enabled in `flowlet_fomo300k` by default,
so training will apply this cache before forming splits.
