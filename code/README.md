# A5P4 code - Transition-aware verification of AI-generated infrastructure security patches

Run (CPU only, ~2 min on 2 cores):

    pip install -r requirements.txt
    python3 run_all.py

`run_all.py` reads `../data/A5P4_ai_security_patches.csv` and writes to `../results/` and `../figures/`.
Everything is seeded (splits use seeds 0-4); only the timing numbers vary between runs.

Pipeline: train-only pre-processing (median imputation + missing flags, winsorising at the 99.5th percentile and
log1p of live connections / LOC, one-hot, z-score) -> nine transition features (Eq. 1) -> proposed DW-kNN-TV
(MI-weighted metric, inverse-distance votes with exponent p, class-prior correction alpha; k, p, alpha tuned on the
validation split) -> baselines (static final-state check, logistic regression, decision tree; with and without
transition features) -> ablation, (k, p) sensitivity, recall@budget, exposure-window regression, paired t-tests.
Identifiers, timestamp, verification_outcome and exposure_window_s are never used as features.

`fig1_flowchart.dot` is the Graphviz source of Fig. 1 (rendered by run_all.py when `dot` is installed).

## Outputs (results/)
| File | Content |
|---|---|
| table1_dataset_summary.csv | Dataset summary (paper Table I) |
| selected_hyperparameters.csv | Hyper-parameters chosen on validation per seed (Table II) |
| table3_main_comparison.csv, main_comparison_mean_std.csv, main_per_seed.csv | Main comparison, mean +- std over 5 seeds (Table III) |
| table4_ablation.csv, ablation_per_seed.csv | Ablation of DW-kNN-TV (Table IV) |
| paired_ttests.csv | Paired t-tests DW-kNN-TV vs each baseline |
| recall_at_budget.csv, recall_at_budget_per_seed.csv | Recall of unsafe patches vs replay budget (Fig. 3) |
| sensitivity_k_p_val_macroF1.csv, sensitivity_k_p_test_macroF1.csv | (k, p) grid, alpha = 0.5 (Fig. 4) |
| confusion_matrix_dwknn_sum5seeds.csv | 5-class confusion matrix (Fig. 5) |
| per_class_detection.csv | Detection rate per hazard class (Fig. 6) |
| mi_feature_weights.csv | Mean MI metric weights w_j |
| exposure_regression*.csv | log exposure-window regression R2 |
| timing.csv | Fit / index rebuild / query time |
| generator_outcome_counts.csv | Outcome counts per generator model |
| metrics.json, run_log.txt | All headline numbers and console log |
| fig*.jpg | JPG copies of all figures (fig7_metric_weights is supplementary, not in the paper) |
