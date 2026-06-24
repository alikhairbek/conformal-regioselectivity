# conformal-regioselectivity# Physics-Informed Conformal Prediction of Aromatic C–H Site-Selectivity

*Guaranteed-coverage prediction sets for electrophilic aromatic substitution (EAS)*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/conformal-regioselectivity/blob/main/notebooks/Conformal_Regioselectivity_Full.ipynb)
[![Made with RDKit](https://img.shields.io/badge/made%20with-RDKit-1f6feb.svg)](https://www.rdkit.org/)
[![Conformal Prediction](https://img.shields.io/badge/uncertainty-conformal%20prediction-009E73.svg)](#method)


---

## Overview

Predicting the **site** of electrophilic aromatic substitution is a foundational task in synthetic and
medicinal chemistry, yet existing quantum-chemical and machine-learning predictors return only a single
most-likely position or an uncalibrated ranking — **with no guarantee** that the experimentally observed
centre is actually captured.

This repository reframes aromatic C–H site-selectivity as a **set-valued prediction problem** and solves it
with **conformal prediction**, a distribution-free framework that converts any score into prediction sets
carrying a user-specified, finite-sample **coverage guarantee**. Because the guarantee holds regardless of the
underlying model, a more physically informative descriptor manifests as a **measurably smaller guaranteed set
at fixed coverage** — a property we use as a quantitative lens on descriptor quality.

The model returns, for every substrate, a **guaranteed set of candidate reactive sites** plus a built-in
**out-of-domain flag**, and it becomes simultaneously **more accurate and more efficient** as physics-informed
electronic descriptors (GFN1-xTB charges, then Fukui *f⁻* indices) are added.

## Highlights

-  **Guaranteed coverage.** Split (inductive) conformal prediction with a deterministic adaptive-prediction-set
  (APS) score yields ≥ 1 − α *any-hit* coverage in finite samples.
-  **Physics-informed descriptors.** A three-stage atom-level representation: topological/Gasteiger → **GFN1-xTB
  atomic charges** → **condensed Fukui *f⁻*** (one extra tight-binding single-point on the radical cation).
-  **More confident *and* more efficient.** Adding electronic descriptors raises top-1 accuracy **and** shrinks
  the guaranteed set, at maintained coverage.
-  **Conditional validity.** Mondrian (class-conditional) calibration restores coverage within sparse and crowded
  substrate strata, and the guarantee **transfers across brominating / iodinating / chlorinating** reagents.
-  **Interpretable.** A leave-one-in ablation identifies the Fukui indices as the single most informative
  electronic descriptors, and Fukui *f⁻* reproduces textbook regiochemistry for archetypal heteroarenes.
-  **Built-in applicability domain.** The conformal credibility score flags out-of-distribution substrates at
  no extra cost.

## Key results

Averaged over 15 calibration/test resamples at a **0.90** target coverage:

| Descriptor stage | # descriptors | Top-1 acc. | Coverage | Mean \|set\| |
|---|:---:|:---:|:---:|:---:|
| **Stage 1** — RDKit (topological + Gasteiger) | 12 | 0.749 | 0.894 | 2.80 |
| **Stage 2** — + GFN1-xTB charges | 17 | 0.818 | 0.906 | 2.74 |
| **Stage 3** — + Fukui *f⁻* | 20 | **0.862** | 0.899 | **2.62** |

Additional findings: Mondrian overall coverage **0.92**; out-of-domain substrates flagged ≈ **3.7 %**; coverage
statistically **homogeneous across reagent classes** (χ² homogeneity *p* = 0.76); most informative electronic
descriptor = **Fukui *f⁻***.

> Dataset: **535** experimental EAS substrates, **604** observed centres (471 single-, 64 multi-centre), median 4
> candidate sites per molecule.

## Repository structure

```
conformal-regioselectivity/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── Conformal_Regioselectivity_Full.ipynb     # end-to-end, runnable on Colab GPU
├── src/
│   └── conformal_regioselectivity_full.py         # script export of the pipeline
├── figures/                                        # 300 dpi, Wong colour-blind-safe palette
│   ├── fig1_validity.png
│   ├── fig2_conditional_coverage.png
│   ├── fig3_setsize.png
│   ├── fig4_efficiency_curve.png
│   ├── fig4b_stage_comparison.png
│   ├── fig5_credibility_ood.png
│   ├── fig6_ablation.png
│   ├── fig7_fukui_validation.png
│   ├── fig8_reagent_robustness.png
│   └── figS1_yield_credibility.png                 # negative result (Supporting Information)
├── results/
│   ├── metrics_coverage.csv
│   ├── metrics_efficiency_curve.csv
│   ├── metrics_stage_comparison.csv
│   ├── metrics_ablation.csv
│   ├── metrics_fukui_validation.csv
│   ├── metrics_reagent_robustness.csv
│   └── summary.json
├── cache/
│   └── xtb_features_cache.pkl                       # cached GFN1-xTB charges + Fukui (exact reproducibility)

```

## Quickstart

### Option A — Google Colab (recommended)

Click the **Open in Colab** badge above, then **Runtime → Run all**. A GPU runtime is recommended
(`Runtime → Change runtime type → GPU`). The notebook installs all dependencies, downloads the dataset,
computes descriptors, runs the conformal pipeline, and regenerates every figure.

### Option B — Local

```bash
git clone https://github.com/USERNAME/conformal-regioselectivity.git
cd conformal-regioselectivity
pip install -r requirements.txt
jupyter notebook notebooks/Conformal_Regioselectivity_Full.ipynb
```

**Requirements:** `rdkit`, `tblite`, `xgboost` (or `lightgbm`), `scikit-learn`, `numpy`, `pandas`,
`scipy`, `matplotlib`. The GFN1-xTB descriptors are cached in `cache/xtb_features_cache.pkl`, so re-runs are
fast; delete the cache to recompute from scratch.

### Configuration

| Flag | Default | Description |
|---|:---:|---|
| `RUN_XTB` | `True` | Compute GFN1-xTB charges + Fukui indices (uses the cache when available). |
| `XTB_MAX_MOLECULES` | `0` | `0` = all molecules; set a small number for a quick smoke test. |
| `ALPHA` | `0.10` | Miscoverage level (target coverage = 1 − α = 0.90). |
| `K_SPLITS` | `15` | Number of calibration/test resamples for the headline metrics. |

## Method

**Problem.** For each molecule, every aromatic C–H is a candidate site. A prediction *set* is deemed to cover the
molecule if it contains **at least one** experimentally observed centre (a multi-label *any-hit* criterion).

**Descriptors (three stages).**
1. **RDKit** — Gasteiger charges, E-state, Crippen logP/MR contributions, ring/heteroatom environment (12 features).
2. **+ GFN1-xTB charges** — self-consistent tight-binding atomic charges from a single-point (via `tblite`),
   assembled into a charge-shell block (5 features).
3. **+ Fukui *f⁻*** — condensed electrophilic Fukui index `f⁻ₖ = qₖ(N−1) − qₖ(N)` from one extra single-point on
   the radical cation at the neutral geometry (3 features).

**Conformal layer.** A gradient-boosted classifier scores each site; scores are renormalized per molecule into a
distribution π. Split-conformal calibration with the **deterministic APS** nonconformity score produces the
guaranteed sets; **Mondrian** calibration equalizes coverage across substrate strata and reagent classes; the
**conformal credibility** of the top-ranked site serves as a distribution-free out-of-domain score.

## Figures

| | |
|---|---|
| `fig1_validity` | Empirical vs. target coverage — finite-sample validity. |
| `fig2_conditional_coverage` | Coverage by substrate stratum; Mondrian restores conditional validity. |
| `fig3_setsize` | Instance-adaptive prediction-set sizes. |
| `fig4_efficiency_curve` | Guaranteed set size vs. cumulative descriptors (both electronic blocks below the RDKit plateau). |
| `fig4b_stage_comparison` | Three-stage accuracy ↑ / set size ↓ comparison. |
| `fig5_credibility_ood` | Applicability domain via conformal credibility. |
| `fig6_ablation` | Leave-one-in descriptor attribution (Fukui dominates). |
| `fig7_fukui_validation` | Fukui *f⁻* reproduces textbook EAS regiochemistry. |
| `fig8_reagent_robustness` | Coverage transfers across electrophile class (*p* = 0.76). |

## Dataset

The benchmark is the openly available EAS regioselectivity set distributed with **RegioSQM20 / RegioML**
(Jensen group): <https://github.com/jensengroup/db-regioselectivity> (MIT License). Please cite the original
dataset papers when using it:

- Ree, N.; Göller, A. H.; Jensen, J. H. *RegioSQM20: Improved Prediction of the Regioselectivity of Electrophilic
  Aromatic Substitutions.* **J. Cheminform.** 2021, 13, 10. <https://doi.org/10.1186/s13321-021-00490-7>
- Ree, N.; Göller, A. H.; Jensen, J. H. *RegioML: Predicting the Regioselectivity of Electrophilic Aromatic
  Substitution Reactions Using Machine Learning.* **Digital Discovery** 2022, 1, 108–114.
  <https://doi.org/10.1039/D1DD00032B>

## Citing this work

A manuscript describing this method is in preparation. Until it appears, please cite the repository:

```bibtex
@software{khairbek_conformal_eas,
  author  = {Khairbek, Ali A.},
  title   = {Physics-Informed Conformal Prediction of Aromatic C--H Site-Selectivity},
  year    = {2026},
  url      = {https://github.com/alikhairbek/conformal-regioselectivity},
  url      = {},
  note    = {Guaranteed-coverage prediction sets for electrophilic aromatic substitution}
}
```

## Key references

- Vovk, V.; Gammerman, A.; Shafer, G. *Algorithmic Learning in a Random World*; Springer, 2005.
- Angelopoulos, A. N.; Bates, S. *Conformal Prediction: A Gentle Introduction.* **Found. Trends Mach. Learn.** 2023, 16, 494–591. <https://doi.org/10.1561/2200000101>
- Romano, Y.; Sesia, M.; Candès, E. J. *Classification with Valid and Adaptive Coverage* (APS). **NeurIPS** 2020.
- Grimme, S.; Bannwarth, C.; Shushkov, P. *GFN1-xTB.* **J. Chem. Theory Comput.** 2017, 13, 1989–2009. <https://doi.org/10.1021/acs.jctc.7b00118>
- Parr, R. G.; Yang, W. *Density Functional Approach to the Frontier-Electron Theory of Chemical Reactivity.* **J. Am. Chem. Soc.** 1984, 106, 4049–4050. <https://doi.org/10.1021/ja00326a036>

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE). The bundled dataset is redistributed under its own
MIT License from the Jensen group repository.

## Acknowledgments & contact

**Ali A. Khairbek** — Department of Chemistry, Chandigarh University, Gharuan, Mohali, Punjab, India
· ORCID [0000-0002-0477-5896](https://orcid.org/0000-0002-0477-5896)

Built with [RDKit](https://www.rdkit.org/), [tblite](https://github.com/tblite/tblite) (GFN1-xTB),
and [XGBoost](https://xgboost.readthedocs.io/). Dataset courtesy of the
[Jensen group](https://github.com/jensengroup/db-regioselectivity).
