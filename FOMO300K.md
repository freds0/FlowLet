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
