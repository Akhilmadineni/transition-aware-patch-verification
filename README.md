# Transition-Aware Verification of AI-Generated Infrastructure Security Patches

This repository contains the experiment code, synthetic dataset, and stored numerical results and figures for **DW-kNN-TV**.

## Study and data

The distance-weighted k-nearest-neighbour transition verifier adds nine transition features, weights the distance metric using mutual information, applies inverse-distance voting and class-prior correction, and prioritises patches for sandbox replay. Baselines include a static final-state check, logistic regression, decision trees and plain kNN.

The dataset is **entirely synthetic**: 12,000 records and 25 columns, generated with seed 2080. It does not contain real credentials, people, companies, IP addresses, or CVEs. See [`data/A5P4_README.md`](data/A5P4_README.md) for the data dictionary and generation assumptions.

The supplied dataset README refers to `gen_author5.py`, but that generator script is not included in the supplied project. The included CSV supports rerunning the experiments; regenerating the raw dataset from scratch is outside this package.

## Repository contents

- `code/run_all.py`: complete experiment pipeline.
- `code/README.md`: method details and output descriptions.
- `requirements.txt`: Python package requirements, copied unchanged from `code/requirements.txt`.
- `data/A5P4_ai_security_patches.csv`: synthetic input dataset.
- `data/A5P4_README.md`: data dictionary and synthetic generation assumptions.
- `results/`: stored result CSVs, JPG figures, and `metrics.json`.

The code uses paths relative to its own location. Keep `code/`, `data/`, and `results/` as sibling directories. Upload these contents at the repository root.

## Requirements

Use Python 3.10 or newer and an isolated virtual environment. Graphviz (`dot`) is optional for regenerating the methodology flowchart. Other plots and numerical results do not require it. Install the system Graphviz program separately when needed; it is not installed by `pip`.

The supplied requirements specify minimum package versions rather than a locked environment. Exact numeric values and runtime can vary with dependency versions, platform, and hardware.

## Reproduce the experiment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/run_all.py
```

On Windows, activate the environment with `.venv\Scripts\activate` before running the same Python commands. If the Graphviz step is required, check that `dot -V` succeeds first.

The complete run takes about 2 minutes on two CPU cores, according to the supplied code README. It rewrites files in `results/` and creates `figures/` with PNG outputs. Existing result files are included for inspection without rerunning model training.

Evaluation uses five stratified random splits (seeds 0–4), each using 60% training, 20% validation and 20% test data. Preprocessing is fitted inside the training data, and validation data select decision thresholds and other tuned settings. Identifiers, the outcome label, and the auxiliary exposure-window outcome do not enter the prediction inputs.

## Stored results

The stored five-seed results report unsafe F1 0.745 and PR-AUC 0.827 for DW-kNN-TV. Unsafe F1 is similar to the strongest feature-augmented baselines; LR with the same transition features has higher five-class macro-F1.

Main outputs are `results/table1_dataset_summary.csv`, `results/table3_main_comparison.csv`, `results/table4_ablation.csv`, and `results/metrics.json`. The remaining CSVs contain per-split metrics and supporting analyses. JPG figures accompany the tables. Timing measurements are environment-dependent.

## Limitations

The hazard generator and transition features encode related hand-designed mechanisms. The replay labels are assumed perfect, and real plans contain richer dependency graphs. Results require validation on operational traces.

The stored results come from synthetic experiments and do not establish production effectiveness. The source project did not supply a fully pinned environment. This package preserves the supplied code and data, with repository documentation added for reproducibility.

## Author and license

Author: Akhil Venkata Sai Madineni. This package is distributed under the [MIT license](LICENSE).

## Citation and archive

Source repository: [https://github.com/Akhilmadineni/transition-aware-patch-verification](https://github.com/Akhilmadineni/transition-aware-patch-verification).

Use [`CITATION.cff`](CITATION.cff) for the software citation. The matching [`.zenodo.json`](.zenodo.json) supplies the archive metadata for release `v1.0.0`. The version DOI will be added after Zenodo completes archiving.
