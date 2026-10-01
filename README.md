# Responsible AI & Model Interpretation

**Alfido Tech Internship - AI Task 4**

Fairness, bias and explainability analysis of a Random Forest income classifier (Adult Income dataset)
using **SHAP** and **LIME**, with bias checks across `sex` and `race` and a comparison of mitigation strategies.

## What's inside

| File | Purpose |
|------|---------|
| `responsible_ai_analysis.ipynb` | Notebook with all interpretation plots and analysis (**deliverable 1**) |
| `analysis.py` | Same analysis as a script (the notebook is generated from it) |
| `make_report.py` | Builds the PDF write-up from the results (**deliverable 2**) |
| `build_notebook.py` | Regenerates the notebook from `analysis.py` |
| `figures/`, `results/` | Plots (`.png`) and `metrics.json` created by the run |
| `requirements.txt` | Dependencies |

## Setup

Python 3.11+ (tested on 3.12). **Internet is needed on the first run** to download the Adult dataset (scikit-learn / OpenML).

```bash
git clone https://github.com/YOUR_USERNAME/responsible-ai-model-interpretation.git
cd responsible-ai-model-interpretation

python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Run

Option A - notebook:

```bash
jupyter notebook responsible_ai_analysis.ipynb      # then Kernel > Restart & Run All, and save
```

Option B - script:

```bash
python analysis.py
```

Check the first lines of the output say `Dataset: Adult Income (OpenML id 1590)`.
If they say "SYNTHETIC", the download failed and the results are not valid.

Build the PDF write-up:

```bash
python make_report.py --name "Your Name" --github "https://github.com/YOUR_USERNAME/responsible-ai-model-interpretation"
```

## Method summary

1. Train a baseline Random Forest on all features (including sex and race, to audit them).
2. Feature importance: impurity-based and permutation importance.
3. SHAP (global summary + local) and LIME (local) explanations for two individuals.
4. Bias audit by sex and race: selection rate, TPR, FPR, demographic-parity difference,
   disparate-impact ratio, equal-opportunity difference; proxy-feature check.
5. Mitigation: (M1) drop sensitive features, (M2) drop + reweighing, (M3) group-specific thresholds.

After installing, you can lock exact versions with `pip freeze > requirements.lock.txt`.
