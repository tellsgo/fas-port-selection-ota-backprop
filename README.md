# Nearest Is Not Safest: Fluid-Antenna Port Selection for Over-the-Air Backpropagation in Urban Multipath

Code and reference results for the paper of the same title.

A terminal with a fluid antenna moves between the forward pass and the backward pass of over-the-air training. It sends the backward signal from the port whose channel should match the forward channel. The paper shows that the port nearest to the forward position reverses a large fraction of the backward gradients in ray-traced streets at a port spacing of 0.9 wavelengths, gives a second-moment certificate for the sign of the mean correlation, and compares deployable selection rules (LoS direction with elevation, survey maps, virtual-source transport with the causal history, learned predictors) with the nearest port.

## Contents

| Folder | What it holds |
|---|---|
| `scripts/` | Everything that runs on a CPU in minutes from the shipped files: tables, figures, certificate, training surrogates, 3GPP CDL check |
| `full/` | The ray-tracing pipeline and the model-fitting scripts that produced the reference results (needs Sionna RT and the scene data, see below) |
| `results/` | Reference results the numbers of the paper come from (JSON, CSV, small NPZ of per-query choices and channels) |
| `data/` | The CDL parameter tables of 3GPP TR 38.901 used by the CDL check |
| `out/` | Created by a run |

## Requirements

Python 3.9 or newer.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the quick and cpu modes (numpy, scipy, pandas, matplotlib, scikit-learn). The full mode additionally needs `requirements-full.txt` (PyTorch for the learned predictors, Sionna RT with its Mitsuba/Dr.Jit backend for ray tracing). No GPU is needed in any mode.

## Reproduce

```
bash run_all.sh quick    # default, about 1 minute on a laptop CPU
bash run_all.sh cpu      # quick + all CPU experiments without ray tracing, about 20 minutes
bash run_all.sh full     # regenerate the ray-traced data and every result, many CPU hours
```

`quick` recomputes Tables I to IV, the bootstrap intervals quoted in the text, Figs. 1 to 5 and the certificate constants from the shipped per-query results, and reruns the training surrogate at 0.9 wavelengths. Every recomputed number is checked against the printed value and reported as PASS or FAIL. `cpu` adds the surrogate at 0.7 wavelengths, the finite-power link training of the extended version and the 3GPP CDL experiment, each checked against its reference JSON (the runs are seeded and reproduce the reference to the last digit on our machines).

Measured on one Apple M5 laptop (single process): certificate 1 s, tables 35 s, figures 5 s, surrogate at one spacing 35 s, finite-power training 75 s, CDL check 17 minutes.

## Results and the paper

Numbering follows the submitted version. The extended version (arXiv) contains further appendix tables, listed at the end.

| In the paper | Produced by | Reference result |
|---|---|---|
| Fig. 1, certificate region; example intervals of Section IV | `scripts/certificate_example.py`, `scripts/make_figures.py` | `results/certificate/certificate_example.json` |
| Table I, pedestrian motion, Chicago, 1000 queries per site | `full/gen_real4.py` (ray tracing) | `results/real4/results_real4_bs{2,1}.json` (bs2 = low site, bs1 = high site) |
| Fig. 2, reversal rate against spacing, 9600 queries per site | `full/gen_real3_data.py` then `full/extract_small.py`; drawn by `scripts/make_figures.py` | `results/extract/fig_bs{2,1}_scat0.npz` |
| Fig. 3, reversal map of the low site | as Fig. 2 | `results/extract/fig_bs2_scat0.npz` |
| Section VIII-D, CDL profiles (38.1, 15.0, 14.3 percent) | `scripts/cdl_check.py` | `results/cdl/results.json` (`pooled_d0.9`: Euc, M2, Orc) |
| Section VIII-D, vehicular displacements, quadratic rule vs Spec | `full/gen_real2_repro.py`; interval recomputed by `scripts/tables.py` | `results/real2_repro/` |
| Table II, unsurveyed positions; intervals of Section VIII-E | `full/real3_model.py`; recomputed by `scripts/tables.py` | `results/real3/test_bs2_scat0_{results.json,choices.npz}`, `results/extract/test_bs2_scat0.npz` |
| Section VIII-E, high site and scattering case | `full/real3_model.py` | `results/real3/test_bs1_scat0_*`, `results/real3/test_bs2_scat03_*` |
| Section VIII-E, interval [-0.3, 11.2] of the history gain | `full/d3_joint_error.py` (error-free level); recomputed by `scripts/tables.py` | `results/d3/d3_results.json` (`levels.none.Map-VSHist`) |
| Table III, five cities without a survey; intervals of Section VIII-F | `full/gen_xscene.py`, `full/lm_chicago.py`, `full/xtest.py`; recomputed by `scripts/tables.py` | `results/xscene/xtest_results.json`, `results/xscene/xtest_per_query.csv.gz` |
| Fig. 4, per-site reversal rate | `scripts/make_figures.py` | `results/xscene/xtest_results.json` (`summary.0.9.by_scene_site`) |
| Table IV, mean clipped alignment | `scripts/tables.py` | `results/e2e/mean_alignment.json` |
| Fig. 5, distribution of the clipped alignment | `scripts/make_figures.py` | as Table II |
| Section VIII-G, digits surrogate (87.8 vs 94.7 to 95.2 percent) | `scripts/e2e_surrogate.py` | `results/e2e/results_E2.json` |
| Extended version: finite-power link training table | `scripts/d1_finite_power.py` | `results/d1/d1_v2_results.json` |
| Extended version: error budget figure and table | `full/d3_joint_error.py` | `results/d3/d3_results.json` |
| Extended version: full CDL table | `scripts/cdl_check.py` | `results/cdl/results.json` |

Rule names in the result files: `Euc` = nearest port, `BSdir`/`BSdir3D`/`LoS3D` = LoS rule, `Cmet` = quadratic rule, `StrongCKM-H0` = map without history selected on validation data, `VS-Hist` (= `StrongCKM-H1` at the full and 10 m surveys) = virtual-source transport fitted to the causal history, `NeuralCKM-H0`/`Learner-H1` and `WM-H0`/`WM-H1` = learned predictors without/with history, `denseSpec` = Spec, `Orc` = best of four candidates, `M2` = bound rule.

## Data

The large ray-traced data sets (about 45 to 65 MB per site for Chicago and 3000 queries per site for the four other cities) are not in the repository. Quick and cpu modes need only the small per-query files in `results/extract/`, which `full/extract_small.py` derives from them.

- **Chicago scene and trajectories.** From the dataset pipeline that accompanies arXiv:2608.10222 (scene `01chicago`: `01chicago.xml` with meshes, `scene_config.json` with the three base stations, `xyz_paths.json` with the trajectories). Obtain it from the release of that work and follow its license; it is not redistributed here. Point `CHICAGO_DIR` to the folder that holds these files.
- **Munich, Etoile, Florence, San Francisco.** The example scenes shipped with Sionna RT (Apache License 2.0), loaded by name in `full/gen_xscene.py`; no download needed.
- **3GPP CDL profiles.** `data/cdl_38901_v18_tables.txt` is an unchanged text extraction of Tables 7.7.1-1 to 7.7.1-5 of 3GPP TR 38.901 V18.0.0 (ETSI TR 138 901 V18.0.0, 2024-05), freely available from ETSI and 3GPP; the copyright of the specification remains with its owners.
- **Digits.** The 8x8 digits data set bundled with scikit-learn (BSD license).

## Notes

- The ray tracer was run at 3.5 GHz with isotropic single antennas, at most five interactions, specular reflection, diffraction and edge diffraction, and seed 7. A fixed seed does not make the solver output identical across runs, so a full rerun reproduces the rates of the paper statistically, not digit by digit. Everything downstream of the ray-traced data (tables, intervals, figures, surrogates) is deterministic.
- `full/real3_model.py` and `full/xtest.py` were each run once on the test split after tuning on the validation split; their outputs are shipped as reference results.
- The quick-mode recomputation of the vehicular interval uses its own bootstrap seed, so it matches the reference [4.6, 11.2] only within 0.6 points; all other intervals are recomputed with the original resampling code and seeds and match exactly.
- The finite-power link training and the error budget appear only in the extended version.
- Not included: the one-off sanity check of Section VIII-B (median of s of 0.999, 0.992 and 0.81 after random displacements of 0.01, 0.05 and 0.25 wavelengths at 50 positions) and the path-count statistics of the scattering case (median 38 and 727 paths). Both were quick interactive checks on the ray-traced data whose scripts were not kept; they do not enter any table or figure.

## License

Code: MIT (see `LICENSE`). Third-party data keep their own terms (see Data).

## Citation

arXiv:2611.02384 (https://arxiv.org/abs/2611.02384). Journal reference to be added.
