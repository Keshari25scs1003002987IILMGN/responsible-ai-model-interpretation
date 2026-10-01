# %% [markdown]
# # Task 4: Responsible AI & Model Interpretation
# **Alfido Tech Internship - AI Track**
#
# **Goal:** analyze fairness, bias and explainability of a trained model using SHAP / LIME and propose mitigation.
#
# **Plan**
# 1. Load the Adult Income dataset (predict whether income is >50K; sensitive attributes: `sex`, `race`)
# 2. Train a baseline Random Forest
# 3. Feature importances (impurity + permutation)
# 4. Local and global explanations with SHAP and LIME
# 5. Bias audit across sensitive groups
# 6. Mitigation experiments and comparison

# %%
import json
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

try:
    get_ipython  # noqa: F821
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
SEED = 42
RESULTS = {}


def save_show(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def show(df):
    if IN_NOTEBOOK:
        from IPython.display import display
        display(df)
    else:
        print(df.to_string())


# %% [markdown]
# ## 1. Load data
# The real **Adult Income** dataset (UCI / OpenML id 1590) is downloaded with scikit-learn.
# If there is no internet connection, a clearly labelled *synthetic* stand-in with the same columns is used
# so that the code can still be tested. **The submitted results must come from the real dataset.**

# %%
def load_adult():
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = ds.frame.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    df = df.rename(columns={"class": "income"})
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), "Unknown").astype(str).str.strip()
    df["income"] = (df["income"] == ">50K").astype(int)
    return df.drop(columns=["fnlwgt", "education"])


def make_synthetic(n=20000, seed=SEED):
    """Offline stand-in with Adult-like columns and a built-in sex/race bias. FOR TESTING ONLY."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Male", "Female"], n, p=[0.67, 0.33])
    race = rng.choice(["White", "Black", "Asian-Pac-Islander", "Other"], n, p=[0.85, 0.09, 0.03, 0.03])
    age = np.clip(rng.normal(38, 13, n), 17, 90).round()
    edu = np.clip(rng.normal(10, 2.5, n), 1, 16).round()
    hours = np.clip(rng.normal(40, 11, n) + (sex == "Male") * 3, 1, 99).round()
    married = rng.random(n) < np.where(sex == "Male", 0.52, 0.28)
    marital = np.where(married, "Married-civ-spouse",
                       rng.choice(["Never-married", "Divorced", "Separated", "Widowed"], n))
    rel = np.where(married, np.where(sex == "Male", "Husband", "Wife"),
                   rng.choice(["Not-in-family", "Own-child", "Unmarried"], n))
    occs = ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair", "Adm-clerical", "Other-service"]
    occ_eff = dict(zip(occs, [1.0, 0.9, 0.2, 0.0, -0.3, -0.9]))
    occ = rng.choice(occs, n)
    work = rng.choice(["Private", "Self-emp", "Gov"], n, p=[0.7, 0.12, 0.18])
    cg = np.where(rng.random(n) < 0.08, rng.exponential(5000, n), 0).round()
    cl = np.where(rng.random(n) < 0.05, rng.exponential(1500, n), 0).round()
    native = rng.choice(["United-States", "Other"], n, p=[0.9, 0.1])
    logit = (-8.6 + 0.04 * age + 0.33 * edu + 0.03 * hours + 1.5 * married + 0.0002 * cg
             + 0.55 * (sex == "Male") + 0.3 * (race == "White") + np.vectorize(occ_eff.get)(occ))
    income = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(dict(age=age, workclass=work, education_num=edu, marital_status=marital,
                             occupation=occ, relationship=rel, race=race, sex=sex, capital_gain=cg,
                             capital_loss=cl, hours_per_week=hours, native_country=native, income=income))


try:
    df = load_adult()
    DATA_SOURCE = "adult"
    print("Dataset: Adult Income (OpenML id 1590)")
except Exception as e:  # no internet etc.
    print("!! Could not download Adult dataset:", repr(e)[:120])
    print("!! Falling back to SYNTHETIC stand-in data - NOT valid for submission.")
    df = make_synthetic()
    DATA_SOURCE = "synthetic"

df["native_country"] = np.where(df["native_country"] == "United-States", "US", "Other")
df["race_group"] = df["race"].where(df["race"].isin(["White", "Black"]), "Other")
RESULTS["data_source"] = DATA_SOURCE
RESULTS["n_rows"] = int(len(df))
RESULTS["positive_rate"] = round(float(df["income"].mean()), 4)
print(df.shape)
show(df.head())

# %% [markdown]
# ### Data balance by sensitive group
# The share of each group and the real rate of `>50K` income in the data. Differences here are **historical bias in the data itself**.

# %%
rows = []
for col in ["sex", "race_group"]:
    for g, sub in df.groupby(col):
        rows.append(dict(attribute=col, group=g, n=len(sub), share=round(len(sub) / len(df), 3),
                         actual_positive_rate=round(sub["income"].mean(), 3)))
balance = pd.DataFrame(rows)
RESULTS["data_balance"] = balance.to_dict("records")
show(balance)

# %% [markdown]
# ## 2. Preprocess and train the baseline model
# One-hot encode categoricals, stratified 70/30 split, Random Forest. The baseline **includes** `sex` and `race`
# so we can audit how much the model relies on them.

# %%
y = df["income"].values
sex = df["sex"]
race_group = df["race_group"]
X = pd.get_dummies(df.drop(columns=["income", "race_group"])).astype(float)
SENS_COLS = [c for c in X.columns if c.startswith("sex_") or c.startswith("race_")]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
sex_train, sex_test = sex.loc[X_train.index], sex.loc[X_test.index]
race_test = race_group.loc[X_test.index]
print("train", X_train.shape, "test", X_test.shape, "| sensitive columns:", SENS_COLS)


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                  random_state=SEED, n_jobs=-1)


model = make_rf().fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
base_metrics = dict(accuracy=accuracy_score(y_test, pred), precision=precision_score(y_test, pred),
                    recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),
                    roc_auc=roc_auc_score(y_test, prob))
RESULTS["baseline_metrics"] = {k: round(float(v), 4) for k, v in base_metrics.items()}
print(RESULTS["baseline_metrics"])

# %% [markdown]
# ## 3. Feature importance
# Two views: **impurity-based** importance (fast, but biased toward high-cardinality/continuous features) and
# **permutation importance** on held-out data (model-agnostic, more reliable).

# %%
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sub_idx = np.random.RandomState(SEED).choice(len(X_test), size=min(3000, len(X_test)), replace=False)
perm = permutation_importance(model, X_test.iloc[sub_idx], y_test[sub_idx], n_repeats=5,
                              scoring="roc_auc", random_state=SEED, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
imp.head(15)[::-1].plot.barh(ax=axes[0], color="#b01e2e")
axes[0].set_title("Impurity-based importance (top 15)")
perm_s.head(15)[::-1].plot.barh(ax=axes[1], color="#2a5d9f")
axes[1].set_title("Permutation importance, drop in ROC-AUC (top 15)")
plt.tight_layout()
save_show(fig, "01_feature_importance")

sens_share = float(imp[SENS_COLS].sum())
RESULTS["top_impurity"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}
RESULTS["top_permutation"] = {k: round(float(v), 4) for k, v in perm_s.head(10).items()}
RESULTS["sensitive_impurity_share"] = round(sens_share, 4)
print(f"Share of impurity importance on sex/race columns: {sens_share:.1%}")

# %% [markdown]
# ## 4. Explainability with SHAP and LIME
# * **SHAP** (TreeExplainer): global summary + local explanations
# * **LIME**: local explanations for the same individuals
#
# Two individuals are explained: a correctly predicted high earner, and (if one exists) a woman with true income
# >50K whom the model rejected (a false negative).

# %%
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print("shap not installed -> run: pip install shap")
try:
    from lime.lime_tabular import LimeTabularExplainer
    HAVE_LIME = True
except ImportError:
    HAVE_LIME = False
    print("lime not installed -> run: pip install lime")
RESULTS["have_shap"], RESULTS["have_lime"] = HAVE_SHAP, HAVE_LIME

pos = np.where((y_test == 1) & (pred == 1))[0]
fn_f = np.where((y_test == 1) & (pred == 0) & (sex_test.values == "Female"))[0]
fn_any = np.where((y_test == 1) & (pred == 0))[0]
idx_a = int(pos[0])
idx_b = int(fn_f[0]) if len(fn_f) else int(fn_any[0])
CASES = {"A (correct high earner)": idx_a,
         "B (" + ("female false negative" if len(fn_f) else "false negative") + ")": idx_b}
for name, i in CASES.items():
    print(name, "| sex:", sex_test.values[i], "| true:", y_test[i], "| P(>50K) =", round(float(prob[i]), 3))
RESULTS["cases"] = {n: dict(sex=str(sex_test.values[i]), true=int(y_test[i]), prob=round(float(prob[i]), 3))
                    for n, i in CASES.items()}


def barh_contrib(ax, names, vals, title):
    order = np.argsort(np.abs(vals))[::-1][:10][::-1]
    ax.barh([names[j] for j in order], [vals[j] for j in order],
            color=["#b01e2e" if vals[j] > 0 else "#2a5d9f" for j in order])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)


# %% [markdown]
# ### 4a. SHAP (global)

# %%
def positive_class(sv):
    """Handle the different return shapes of TreeExplainer across shap versions."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


if HAVE_SHAP:
    explainer = shap.TreeExplainer(model)
    Xs = X_test.sample(500, random_state=SEED)
    sv = positive_class(explainer.shap_values(Xs))

    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=15)
    save_show(plt.gcf(), "02_shap_summary_beeswarm")

    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=15)
    save_show(plt.gcf(), "03_shap_bar")

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    RESULTS["top_shap"] = {k: round(float(v), 4) for k, v in mean_abs.head(10).items()}
    RESULTS["sensitive_shap_share"] = round(float(mean_abs[SENS_COLS].sum() / mean_abs.sum()), 4)
    print("Share of mean |SHAP| on sex/race columns:", f"{RESULTS['sensitive_shap_share']:.1%}")
    show(mean_abs.head(10).round(4).to_frame("mean |SHAP|"))

# %% [markdown]
# ### 4b. SHAP (local) and LIME (local)

# %%
if HAVE_SHAP:
    ev = np.atleast_1d(explainer.expected_value)
    base = float(ev[1] if len(ev) > 1 else ev[0])
    sv_cases = positive_class(explainer.shap_values(X_test.iloc[list(CASES.values())]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, _), row in zip(axes, CASES.items(), sv_cases):
        barh_contrib(ax, list(X.columns), row, f"SHAP - case {name}\n(red pushes toward >50K)")
    plt.tight_layout()
    save_show(fig, "04_shap_local")
    RESULTS["shap_base_value"] = round(base, 4)

if HAVE_LIME:
    lime_explainer = LimeTabularExplainer(
        X_train.values, feature_names=list(X.columns), class_names=["<=50K", ">50K"],
        mode="classification", discretize_continuous=True, random_state=SEED)
    predict_fn = lambda a: model.predict_proba(pd.DataFrame(a, columns=X.columns))  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    RESULTS["lime"] = {}
    for ax, (name, i) in zip(axes, CASES.items()):
        exp = lime_explainer.explain_instance(X_test.values[i], predict_fn, num_features=10, num_samples=3000)
        items = exp.as_list()
        barh_contrib(ax, [t[0] for t in items], np.array([t[1] for t in items]),
                     f"LIME - case {name}\n(red supports >50K)")
        RESULTS["lime"][name] = [[t[0], round(float(t[1]), 4)] for t in items[:5]]
    plt.tight_layout()
    save_show(fig, "05_lime_local")

# %% [markdown]
# ## 5. Bias audit
# Metrics per group on the test set:
# * **Selection rate** - share predicted >50K (demographic parity)
# * **TPR** - share of true high earners that are found (equal opportunity)
# * **FPR** - share of true low earners wrongly predicted high (with TPR: equalized odds)
#
# Summary measures: demographic-parity difference (max-min selection rate), disparate-impact ratio (min/max;
# values < 0.8 fail the common "80% rule"), equal-opportunity difference (TPR gap).

# %%
def group_metrics(y_true, y_pred, groups):
    rows = []
    for g in sorted(pd.Series(groups).unique()):
        m = np.asarray(groups) == g
        yt, yp = y_true[m], y_pred[m]
        rows.append(dict(group=g, n=int(m.sum()), selection_rate=yp.mean(),
                         tpr=yp[yt == 1].mean() if (yt == 1).any() else np.nan,
                         fpr=yp[yt == 0].mean() if (yt == 0).any() else np.nan,
                         accuracy=(yt == yp).mean()))
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(gm):
    return dict(dp_diff=float(gm.selection_rate.max() - gm.selection_rate.min()),
                di_ratio=float(gm.selection_rate.min() / gm.selection_rate.max()) if gm.selection_rate.max() > 0 else np.nan,
                eo_diff=float(gm.tpr.max() - gm.tpr.min()),
                fpr_diff=float(gm.fpr.max() - gm.fpr.min()))


gm_sex = group_metrics(y_test, pred, sex_test.values)
gm_race = group_metrics(y_test, pred, race_test.values)
fs_sex, fs_race = fairness_summary(gm_sex), fairness_summary(gm_race)
RESULTS["baseline_group_sex"] = gm_sex.round(4).reset_index().to_dict("records")
RESULTS["baseline_group_race"] = gm_race.round(4).reset_index().to_dict("records")
RESULTS["baseline_fairness_sex"] = {k: round(v, 4) for k, v in fs_sex.items()}
RESULTS["baseline_fairness_race"] = {k: round(v, 4) for k, v in fs_race.items()}
print("== By sex =="); show(gm_sex.round(3)); print(RESULTS["baseline_fairness_sex"])
print("== By race =="); show(gm_race.round(3)); print(RESULTS["baseline_fairness_race"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, gm, t in zip(axes, [gm_sex, gm_race], ["Sex", "Race group"]):
    gm[["selection_rate", "tpr", "fpr"]].plot.bar(ax=ax, color=["#b01e2e", "#2a5d9f", "#999999"], rot=0)
    ax.set_title(f"Baseline model - rates by {t}")
    ax.set_ylim(0, 1)
plt.tight_layout()
save_show(fig, "06_bias_by_group")

# %% [markdown]
# ### Proxy check
# Removing `sex` from the inputs does not guarantee fairness if other features reveal it. We test how well the
# remaining features can predict `sex` (ROC-AUC 0.5 = no leakage, 1.0 = perfect leakage).

# %%
X_nosens_train, X_nosens_test = X_train.drop(columns=SENS_COLS), X_test.drop(columns=SENS_COLS)
proxy = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED, n_jobs=-1)
proxy.fit(X_nosens_train, (sex_train.values == "Male").astype(int))
proxy_auc = roc_auc_score((sex_test.values == "Male").astype(int), proxy.predict_proba(X_nosens_test)[:, 1])
proxy_imp = pd.Series(proxy.feature_importances_, index=X_nosens_train.columns).sort_values(ascending=False)
RESULTS["proxy_auc"] = round(float(proxy_auc), 4)
RESULTS["proxy_top_features"] = {k: round(float(v), 4) for k, v in proxy_imp.head(5).items()}
print(f"AUC of predicting sex from the remaining features: {proxy_auc:.3f}")
show(proxy_imp.head(5).round(3).to_frame("importance"))

# %% [markdown]
# ## 6. Mitigation experiments
# | Strategy | Stage | Idea |
# |---|---|---|
# | M0 Baseline | - | all features |
# | M1 Drop sensitive features | pre-processing | "fairness through unawareness" |
# | M2 Drop sensitive + Reweighing | pre-processing | Kamiran & Calders: weight each (group, label) cell so group and label look independent |
# | M3 Group thresholds | post-processing | separate decision thresholds per sex, tuned on a validation split to equalize TPR |

# %%
def evaluate(name, y_pred, y_prob=None):
    gm = group_metrics(y_test, y_pred, sex_test.values)
    fs = fairness_summary(gm)
    return dict(model=name, accuracy=accuracy_score(y_test, y_pred), f1=f1_score(y_test, y_pred),
                roc_auc=roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan, **fs)


results = [evaluate("M0 Baseline", pred, prob)]

# M1
m1 = make_rf().fit(X_nosens_train, y_train)
p1 = m1.predict_proba(X_nosens_test)[:, 1]
results.append(evaluate("M1 Drop sensitive", (p1 >= 0.5).astype(int), p1))

# M2 reweighing
w = np.ones(len(y_train))
for s in sex_train.unique():
    for lab in (0, 1):
        cell = (sex_train.values == s) & (y_train == lab)
        if cell.any():
            w[cell] = (sex_train.values == s).mean() * (y_train == lab).mean() / cell.mean()
m2 = make_rf().fit(X_nosens_train, y_train, sample_weight=w)
p2 = m2.predict_proba(X_nosens_test)[:, 1]
results.append(evaluate("M2 Drop + Reweighing", (p2 >= 0.5).astype(int), p2))

# M3 group-specific thresholds (post-processing), thresholds chosen on a validation split
Xa, Xv, ya, yv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
m3 = make_rf().fit(Xa, ya)
pv, pt = m3.predict_proba(Xv)[:, 1], m3.predict_proba(X_test)[:, 1]
sex_v = sex_train.loc[Xv.index].values
target_tpr = ((pv >= 0.5) & (yv == 1)).sum() / (yv == 1).sum()
grid = np.arange(0.05, 0.951, 0.01)
thr = {}
for s in np.unique(sex_v):
    mm = (sex_v == s) & (yv == 1)
    tprs = np.array([(pv[mm] >= t).mean() for t in grid])
    thr[s] = float(grid[np.argmin(np.abs(tprs - target_tpr))])
pred3 = np.array([int(p >= thr[s]) for p, s in zip(pt, sex_test.values)])
results.append(evaluate("M3 Group thresholds", pred3, pt))
RESULTS["m3_thresholds"] = {k: round(v, 2) for k, v in thr.items()}
print("Group thresholds:", RESULTS["m3_thresholds"])

mit = pd.DataFrame(results).set_index("model")
RESULTS["mitigation"] = mit.round(4).reset_index().to_dict("records")
show(mit.round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
mit["accuracy"].plot.bar(ax=axes[0], color="#2a5d9f", rot=20); axes[0].set_title("Accuracy"); axes[0].set_ylim(0.6, 1)
mit["dp_diff"].plot.bar(ax=axes[1], color="#b01e2e", rot=20); axes[1].set_title("Demographic-parity diff (sex) - lower is fairer")
mit["eo_diff"].plot.bar(ax=axes[2], color="#e08a1e", rot=20); axes[2].set_title("Equal-opportunity diff (sex) - lower is fairer")
for a in axes: a.set_xlabel("")
plt.tight_layout()
save_show(fig, "07_mitigation_comparison")

# %% [markdown]
# ## 7. Save results
# `results/metrics.json` and `figures/*.png` are used by `make_report.py` to build the PDF write-up.

# %%
with open("results/metrics.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Saved results/metrics.json and figures:", sorted(os.listdir("figures")))
