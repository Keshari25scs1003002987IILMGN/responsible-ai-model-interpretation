"""Build the PDF write-up from results/metrics.json and figures/*.png.
Usage: python make_report.py --name "Your Name" --github "https://github.com/you/responsible-ai-model-interpretation"
"""
import argparse, json, os
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="YOUR NAME")
ap.add_argument("--github", default="https://github.com/YOUR_USERNAME/responsible-ai-model-interpretation")
ap.add_argument("--out", default="Task4_Responsible_AI_Report.pdf")
args = ap.parse_args()

R = json.load(open("results/metrics.json"))
synthetic = R["data_source"] != "adult"
RED = colors.HexColor("#b01e2e")

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.2, leading=14.5)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=RED, fontSize=15, spaceBefore=12)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8)
title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, textColor=RED)
cap = ParagraphStyle("cap", parent=body, fontSize=8.5, textColor=colors.grey, alignment=1)
warn = ParagraphStyle("w", parent=body, textColor=colors.white, backColor=RED, borderPadding=6, leading=14)

P = lambda t: Paragraph(t, body)
pct = lambda v: f"{v*100:.1f}%"


def bl(items):
    return [Paragraph("&bull; " + i, ParagraphStyle("l", parent=body, leftIndent=14, firstLineIndent=-8)) for i in items]


def table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), RED), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(st))
    return t


def fig(name, caption, width=16.5 * cm):
    path = f"figures/{name}.png"
    if not os.path.exists(path):
        return []
    w, h = PILImage.open(path).size
    return [KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(caption, cap), Spacer(1, 6)])]


s = []
# ---------- cover
s += [Spacer(1, 2.2 * cm), Paragraph("Task 4: Responsible AI &amp; Model Interpretation", title),
      Paragraph("Fairness, bias and explainability analysis with SHAP / LIME", ParagraphStyle(
          "x", parent=body, alignment=1, fontSize=13)), Spacer(1, 1 * cm)]
if synthetic:
    s += [Paragraph("<b>DRAFT - NOT FOR SUBMISSION.</b> These numbers come from a synthetic stand-in dataset "
                    "(no internet was available). Re-run <font face='Courier'>python analysis.py</font> on the real "
                    "Adult dataset, then re-run <font face='Courier'>make_report.py</font>.", warn), Spacer(1, 0.6 * cm)]
if not (R["have_shap"] and R["have_lime"]):
    s += [Paragraph("<b>NOTE:</b> SHAP and/or LIME were not installed for this run, so those sections are empty. "
                    "Run <font face='Courier'>pip install -r requirements.txt</font> and re-run.", warn), Spacer(1, 0.6 * cm)]
s += [table([["Intern name", args.name], ["Task", "4 - Responsible AI & Model Interpretation"],
             ["GitHub repository", args.github],
             ["Dataset", "Adult Income (UCI / OpenML 1590)" if not synthetic else "SYNTHETIC stand-in (draft)"],
             ["Model", "Random Forest (200 trees, max depth 12)"], ["Date", "01-10-2026"]],
            widths=[4 * cm, 12.5 * cm], header=False), PageBreak()]

# ---------- 1 objective & data
s += [Paragraph("1. Objective and Dataset", h1),
      P("The goal is to analyze the <b>fairness, bias and explainability</b> of a classification model with SHAP and LIME, "
        "check for bias across sensitive groups, and propose practical mitigation steps."),
      Spacer(1, 4),
      P(f"The model predicts whether a person earns more than 50K per year from census attributes. The dataset has "
        f"<b>{R['n_rows']:,}</b> rows, of which <b>{pct(R['positive_rate'])}</b> are in the positive (&gt;50K) class. "
        f"The sensitive attributes are <b>sex</b> and <b>race</b> (race grouped into White / Black / Other because some "
        f"categories are very small).")]
rows = [["Attribute", "Group", "n", "Share", "Actual &gt;50K rate"]] if False else [["Attribute", "Group", "n", "Share", "Actual >50K rate"]]
rows += [[d["attribute"], d["group"], f"{d['n']:,}", pct(d["share"]), pct(d["actual_positive_rate"])] for d in R["data_balance"]]
s += [Spacer(1, 6), table(rows, widths=[3.5 * cm, 3.5 * cm, 2.5 * cm, 3 * cm, 4 * cm]), Spacer(1, 4),
      P("Differences in the real positive rate between groups already show <b>historical bias in the data</b>; "
        "a model trained on it can learn and reproduce that pattern.")]

# ---------- 2 model
m = R["baseline_metrics"]
s += [Paragraph("2. Baseline Model", h1),
      P("A stratified 70/30 train/test split was used (random_state=42), categorical features were one-hot encoded, and a "
        "Random Forest was trained on <b>all</b> features, including sex and race, so that their influence can be audited."),
      Spacer(1, 4),
      table([["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
             [f"{m['accuracy']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}"]])]

# ---------- 3 importance
top_imp = ", ".join(f"<font face='Courier'>{k}</font>" for k in list(R["top_permutation"])[:5])
s += [Paragraph("3. Feature Importance", h1),
      P(f"Two measures were computed: impurity-based importance and permutation importance on held-out data. "
        f"The five features with the largest permutation importance are: {top_imp}. "
        f"The sex and race columns together hold <b>{pct(R['sensitive_impurity_share'])}</b> of the impurity importance."),
      Spacer(1, 4)] + fig("01_feature_importance", "Figure 1 - Feature importance (impurity vs permutation)")

# ---------- 4 explainability
s += [Paragraph("4. Explainability: SHAP and LIME", h1)]
if R["have_shap"]:
    top_shap = ", ".join(f"<font face='Courier'>{k}</font>" for k in list(R["top_shap"])[:5])
    s += [Paragraph("4.1 SHAP - global", h2),
          P(f"SHAP (TreeExplainer) was computed on a 500-row test sample. The features with the largest mean |SHAP| are "
            f"{top_shap}. The sex and race columns account for <b>{pct(R['sensitive_shap_share'])}</b> of the total mean |SHAP|. "
            f"In the beeswarm plot, each dot is one person: horizontal position shows the effect on the prediction and "
            f"colour shows the feature value.")]
    s += fig("02_shap_summary_beeswarm", "Figure 2 - SHAP summary (beeswarm)", 13 * cm)
    s += fig("03_shap_bar", "Figure 3 - Mean |SHAP| per feature", 13 * cm)
    s += [Paragraph("4.2 SHAP - local", h2)] + fig("04_shap_local", "Figure 4 - SHAP explanations for two individuals")
if R["have_lime"]:
    s += [Paragraph("4.3 LIME - local", h2),
          P("LIME fits a simple local surrogate model around one prediction, so it shows which feature ranges pushed that "
            "single prediction up or down. The same two individuals as in the SHAP section were explained.")]
    s += fig("05_lime_local", "Figure 5 - LIME explanations for the same two individuals")
    s += [P("<b>SHAP vs LIME:</b> SHAP has consistent additive attributions and is exact for tree models; LIME is model-agnostic "
            "but depends on random sampling and the chosen kernel, so its explanations can vary between runs.")]
if not (R["have_shap"] or R["have_lime"]):
    s += [P("<i>SHAP and LIME were not available in this run.</i>")]
case_rows = [["Case", "Sex", "True label", "Predicted P(>50K)"]] + [
    [k, v["sex"], ">50K" if v["true"] else "<=50K", f"{v['prob']:.3f}"] for k, v in R["cases"].items()]
s += [Spacer(1, 4), table(case_rows, widths=[6 * cm, 3 * cm, 3.5 * cm, 4 * cm])]

# ---------- 5 bias audit
fs, fr = R["baseline_fairness_sex"], R["baseline_fairness_race"]
gs = {d["group"]: d for d in R["baseline_group_sex"]}
s += [Paragraph("5. Bias Audit", h1),
      P("Metrics per group on the test set: <b>selection rate</b> (share predicted &gt;50K, demographic parity), "
        "<b>TPR</b> (share of true high earners found, equal opportunity) and <b>FPR</b>. The disparate-impact ratio is the "
        "lowest selection rate divided by the highest; values below 0.8 fail the common '80% rule'."), Spacer(1, 4)]
for title_, key in [("By sex", "baseline_group_sex"), ("By race group", "baseline_group_race")]:
    rows = [["Group", "n", "Selection rate", "TPR", "FPR", "Accuracy"]] + [
        [d["group"], f"{d['n']:,}", pct(d["selection_rate"]), pct(d["tpr"]), pct(d["fpr"]), pct(d["accuracy"])] for d in R[key]]
    s += [Paragraph(title_, h2), table(rows)]
s += [Spacer(1, 6),
      table([["Sensitive attribute", "DP difference", "Disparate-impact ratio", "Equal-opp. difference", "FPR difference"],
             ["sex", f"{fs['dp_diff']:.3f}", f"{fs['di_ratio']:.3f}", f"{fs['eo_diff']:.3f}", f"{fs['fpr_diff']:.3f}"],
             ["race group", f"{fr['dp_diff']:.3f}", f"{fr['di_ratio']:.3f}", f"{fr['eo_diff']:.3f}", f"{fr['fpr_diff']:.3f}"]]),
      Spacer(1, 6)]
verdict = "<b>fails</b>" if fs["di_ratio"] < 0.8 else "<b>passes</b>"
s += [P(f"For sex, the baseline model's disparate-impact ratio is {fs['di_ratio']:.2f}, which {verdict} the 80% rule, and the "
        f"true-positive-rate gap is {pct(fs['eo_diff'])}. For race groups the ratio is {fr['di_ratio']:.2f} and the TPR gap is "
        f"{pct(fr['eo_diff'])}."), Spacer(1, 4)]
s += fig("06_bias_by_group", "Figure 6 - Selection rate, TPR and FPR by sex and race group")

s += [KeepTogether([Paragraph("Proxy features", h2),
      P(f"Dropping the sensitive columns does not guarantee fairness if other features reveal them. Using only the remaining "
        f"features, a model can predict <b>sex</b> with ROC-AUC <b>{R['proxy_auc']:.3f}</b> (0.5 = no leakage). The most "
        f"informative proxies are " + ", ".join(f"<font face='Courier'>{k}</font>" for k in R["proxy_top_features"]) + ".")])]

# ---------- 6 mitigation
mit = R["mitigation"]
base = mit[0]
rows = [["Strategy", "Accuracy", "F1", "ROC-AUC", "DP diff", "DI ratio", "EO diff"]] + [
    [d["model"], f"{d['accuracy']:.3f}", f"{d['f1']:.3f}", f"{d['roc_auc']:.3f}", f"{d['dp_diff']:.3f}",
     f"{d['di_ratio']:.3f}", f"{d['eo_diff']:.3f}"] for d in mit]
best = min(mit[1:], key=lambda d: d["dp_diff"])
s += [Paragraph("6. Mitigation Experiments (fairness measured on sex)", h1),
      table([["Strategy", "Stage", "Description"],
             ["M0 Baseline", "-", "All features"],
             ["M1 Drop sensitive", "Pre-processing", "Remove sex and race columns (unawareness)"],
             ["M2 Drop + Reweighing", "Pre-processing", "Kamiran-Calders sample weights per (sex, label) cell"],
             ["M3 Group thresholds", "Post-processing",
              Paragraph(f"Per-sex decision thresholds {R['m3_thresholds']} tuned on a validation split to equalize TPR", ParagraphStyle("c", parent=body, fontSize=8.5, leading=11))]],
            widths=[4 * cm, 3 * cm, 9.5 * cm]), Spacer(1, 8), table(rows), Spacer(1, 6),
      P(f"Compared with the baseline, the strategy with the smallest demographic-parity gap is <b>{best['model']}</b>: "
        f"the gap changes from {base['dp_diff']:.3f} to {best['dp_diff']:.3f} and the equal-opportunity gap from "
        f"{base['eo_diff']:.3f} to {best['eo_diff']:.3f}, while accuracy changes from {base['accuracy']:.3f} to "
        f"{best['accuracy']:.3f}. Note that demographic parity and equal opportunity can conflict when the real base rates "
        f"differ between groups, so the metric to prioritise is a policy decision, not only a technical one."),
      Spacer(1, 4)] + fig("07_mitigation_comparison", "Figure 7 - Accuracy and fairness gaps per strategy")

# ---------- 7 recommendations
s += [Paragraph("7. Recommended Mitigation Steps", h1)]
s += bl(["<b>Audit the data first:</b> check group representation and label rates, and collect more data for under-represented groups.",
         "<b>Do not rely on removing sensitive columns:</b> proxy features (such as relationship or marital status) still carry the signal; test for proxies as in Section 5.",
         "<b>Pre-processing:</b> use reweighing or resampling so that group and label are less correlated in the training data.",
         "<b>Post-processing:</b> use group-specific thresholds when equal opportunity is the goal. Check that using group-dependent decisions is legally and ethically acceptable for the use case.",
         "<b>Explain and document:</b> publish a model card with SHAP/LIME based explanations, intended use, known limitations and per-group metrics.",
         "<b>Monitor in production:</b> track group-wise selection rate and error rates over time and re-audit when data drifts.",
         "<b>Keep a human in the loop</b> for high-stakes decisions and provide a way to appeal."])
s += [Paragraph("8. Limitations", h1)]
s += bl(["Fairness metrics were computed on one train/test split; no confidence intervals are reported, so small gaps for small groups (for example race 'Other') should be read with care.",
         "The dataset is a 1994 US census extract with binary sex labels; findings do not transfer directly to other populations.",
         "SHAP and LIME explain the model's behaviour, not causal relationships in the real world.",
         "Only sex was targeted by the mitigation strategies; race was audited but not mitigated."])
s += [Paragraph("9. How to Reproduce", h1),
      P("<font face='Courier'>pip install -r requirements.txt</font><br/>"
        "<font face='Courier'>python analysis.py</font> (or open <font face='Courier'>responsible_ai_analysis.ipynb</font> and Run All)<br/>"
        "<font face='Courier'>python make_report.py --name \"Your Name\" --github \"&lt;repo url&gt;\"</font>")]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(RED if synthetic else colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Alfido Tech Internship - Task 4: Responsible AI" + ("   |   DRAFT: SYNTHETIC DATA" if synthetic else ""))
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()


SimpleDocTemplate(args.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="Task 4 - Responsible AI & Model Interpretation").build(s, onFirstPage=footer, onLaterPages=footer)
print("Wrote", args.out)
