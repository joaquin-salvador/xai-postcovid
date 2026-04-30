import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
EXPLAINERS_DIR = os.path.join(DATA_DIR, "explainers")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

# Feature metadata defaults
DEFAULT_CATEGORY_LABELS = {
    "Sex_Male": {0: "Female",
                 1: "Male"},
    "Is_Married": {0: "Unmarried",
                   1: "Married"},
    "Has_Child": {0: "No Children",
                  1: "Has Children"},
    "Income": {0: "Missing",
               1: "<2.0M JPY",
               2: "2.0-3.9M JPY",
               3: "4.0-5.9M JPY",
               4: "6.0-7.9M JPY",
               5: ">=8.0M JPY"},
    "Income_Missing": {0: "Income Reported",
                       1: "Income Missing"},
    "Job_Employed": {0: "No",
                     1: "Yes"},
    "Job_Homemaker": {0: "No",
                      1: "Yes"},
    "Job_Student": {0: "No",
                    1: "Yes"},
    "Job_Unemployed": {0: "No",
                       1: "Yes"},
    "Job_Other": {0: "No",
                  1: "Yes"},
}

DEFAULT_DISPLAY_NAMES = {
    "Sex_Male": "Sex",
    "Age": "Age",
    "Is_Married": "Marital Status",
    "Has_Child": "Has Children",
    "Income": "Household Income",
    "Income_Missing": "Income Missing",
    "Job_Employed": "Employed",
    "Job_Homemaker": "Homemaker",
    "Job_Student": "Student",
    "Job_Unemployed": "Unemployed",
    "Job_Other": "Other Job",
    "Activity": "Social/Physical Activity",
    "Exercise": "Exercise Frequency",
    "Healthy_Diet": "Healthy Diet",
    "Healthy_Sleep": "Healthy Sleep",
    "Interaction_Offline": "Offline Social Interaction",
    "Interaction_Online": "Online Social Interaction",
    "Altruistic": "Altruistic Behavior",
    "Frustration": "Frustration Level",
    "Optimism": "Optimism Level",
    "Covid_Anxiety": "COVID Anxiety",
    "Covid_Sleepless": "COVID-related Sleeplessness",
    "Deterioration_Economy": "Economic Deterioration",
    "Deterioration_Interact": "Social Interaction Deterioration",
    "Difficulty_Living": "Difficulty Living",
    "Difficulty_Work": "Difficulty Working",
}

DEFAULT_LIKERT_FEATURES = [
    "Activity",
    "Exercise",
    "Healthy_Diet",
    "Healthy_Sleep",
    "Interaction_Offline",
    "Interaction_Online",
    "Altruistic",
    "Frustration",
    "Optimism",
    "Covid_Anxiety",
    "Covid_Sleepless",
    "Deterioration_Economy",
    "Deterioration_Interact",
    "Difficulty_Living",
    "Difficulty_Work"
]

BINARY_FEATURES = {
    "Sex_Male",
    "Is_Married",
    "Has_Child",
    "Income_Missing",
    "Job_Employed",
    "Job_Homemaker",
    "Job_Student",
    "Job_Unemployed",
    "Job_Other",
}

JOB_FEATURES = ["Job_Employed",
                "Job_Homemaker",
                "Job_Student",
                "Job_Unemployed",
                "Job_Other"]
JOB_OPTIONS = {
    "Job_Employed": "Employed",
    "Job_Homemaker": "Homemaker",
    "Job_Student": "Student",
    "Job_Unemployed": "Unemployed",
    "Job_Other": "Other",
}


# Lazy imports

def _import_shap():
    import shap
    return shap

def _import_matplotlib():
    import matplotlib.pyplot as plt
    return plt

def _import_plotly():
    import plotly.express as px
    import plotly.figure_factory as ff
    return px, ff


# Helpers

def get_display_name(feat, display_names):
    return display_names.get(feat, feat.replace("_", " "))

def format_feature_value(feat, value, category_labels, likert_features):
    if feat in category_labels:
        int_val = int(round(value))
        return category_labels[feat].get(int_val, str(int_val))
    elif feat in likert_features:
        return f"{value:.1f} / 7"
    elif feat == "Age":
        return f"{value:.0f} years"
    return f"{value:.2f}"

def build_person_label(idx, row, class_names, precomputed_preds=None):
    """Build a human-readable label for a person selector dropdown."""
    age = row.get("Age", None)
    sex = row.get("Sex_Male", None)
    age_str = f"{int(age)}yo" if age is not None and not np.isnan(age) else ""
    sex_str = ("M" if int(sex) == 1 else "F") if sex is not None and not np.isnan(sex) else ""
    pred_str = ""
    if precomputed_preds is not None:
        pred_str = f" — {class_names[int(precomputed_preds['y_pred'][idx])]}"
    demo = ", ".join(p for p in [sex_str, age_str] if p)
    return f"Person {idx + 1} ({demo}{pred_str})"

def get_prediction(model, X, idx, precomputed_preds=None):
    """Get prediction and probabilities for a sample, using precomputed if available."""
    if precomputed_preds is not None and idx < len(precomputed_preds["y_pred"]):
        pred = int(precomputed_preds["y_pred"][idx])
        prob_high = float(precomputed_preds["y_prob"][idx])
        return pred, np.array([1.0 - prob_high, prob_high])
    pred = model.predict(X.iloc[[idx]])[0]
    prob = model.predict_proba(X.iloc[[idx]])[0]
    return int(pred), prob

def build_person_options(X, class_names, precomputed_preds=None):
    """Build person index list and label dict for a selectbox."""
    options = list(range(len(X)))
    labels = {i: build_person_label(i, X.iloc[i], class_names, precomputed_preds)
              for i in options}
    return options, labels

def build_profile_df(X_row, features, display_names, category_labels,
                     likert_features, include_raw=False):
    """Build a profile DataFrame for display."""
    rows = []
    for feat in features:
        val = X_row[feat]
        entry = {
            "Feature": get_display_name(feat, display_names),
            "Value": format_feature_value(feat, val, category_labels, likert_features),
        }
        if include_raw:
            entry["Raw"] = f"{val:.2f}"
        rows.append(entry)
    return pd.DataFrame(rows)


# Data Loading

@st.cache_resource
def load_artifacts():
    """Load all saved model and explainer artifacts."""
    import joblib

    # Use surrogate model as the primary model (CPU-friendly)
    model_path = os.path.join(MODELS_DIR, "surrogate_lgbm.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"surrogate_lgbm.joblib not found in {MODELS_DIR}. "
            "Make sure you exported it from your notebook."
        )

    model = joblib.load(model_path)

    # Keep surrogate reference
    surrogate = model

    X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()

    with open(os.path.join(EXPLAINERS_DIR, "shap_values_test.pkl"), "rb") as f:
        shap_values_test = pickle.load(f)

    # Normalize: ensure list-of-2D format [class0, class1]
    if isinstance(shap_values_test, np.ndarray) and shap_values_test.ndim == 3:
        shap_values_test = [shap_values_test[:, :, i]
                            for i in range(shap_values_test.shape[2])]

    with open(os.path.join(EXPLAINERS_DIR, "shap_expected_value.pkl"), "rb") as f:
        shap_expected_value = np.array(pickle.load(f)).flatten()

    x_explain_path = os.path.join(ARTIFACTS_DIR, "X_explain.csv")
    if os.path.exists(x_explain_path):
        X_explain = pd.read_csv(x_explain_path)
    else:
        n_explained = np.array(shap_values_test[0]).shape[0]
        X_explain = X_test.iloc[:n_explained].copy()

    with open(os.path.join(EXPLAINERS_DIR, "feature_info.pkl"), "rb") as f:
        feature_info = pickle.load(f)

    preds_path = os.path.join(ARTIFACTS_DIR, "test_predictions.pkl")
    precomputed_preds = None
    if os.path.exists(preds_path):
        with open(preds_path, "rb") as f:
            precomputed_preds = pickle.load(f)

    def _load_cf_dict(filenames):
        for cf_name in filenames:
            cf_path = os.path.join(EXPLAINERS_DIR, cf_name)
            if not os.path.exists(cf_path):
                continue
            with open(cf_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                return loaded
            if isinstance(loaded, list):
                return {item["sample_idx"]: item["counterfactuals"] for item in loaded}
        return None

    precomputed_cfs = _load_cf_dict(
        ["counterfactual_results.pkl", "dice_results.pkl"]
    )
    precomputed_cfs_limited = _load_cf_dict(
        ["counterfactual_results_limited.pkl", "dice_results_limited.pkl"]
    )

    return (model, surrogate, X_train, X_test, X_explain, y_train, y_test,
            shap_values_test, shap_expected_value, feature_info,
            precomputed_preds, precomputed_cfs, precomputed_cfs_limited)


# Model Overview Page

def render_overview(model, X_test, y_test, class_names, precomputed_preds):
    px, ff = _import_plotly()
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    st.title("📊 Model Overview")

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.info(
            "This dashboard explains a machine learning model that predicts loneliness levels "
            "based on the UCLA Loneliness Scale. A score of 22 or above indicates **High Loneliness**. "
            "The model uses demographic information, health behaviors, and COVID-related impacts "
            "to make predictions."
        )

    st.subheader("About the Study")
    st.markdown(
        "This study investigates **post-COVID loneliness prediction** using explainable AI. "
        "It draws on longitudinal survey data from **2,659 Japanese respondents** across four "
        "waves (2020--2024), capturing socio-demographic attributes, coping behaviors, "
        "COVID-19 impact measures, and psychological scales including the **UCLA Loneliness "
        "Scale (UCLA-LS3)**."
    )

    st.subheader("Model Architecture & Pipeline")
    st.markdown(
        "**1. Data Preprocessing**\n"
        "- Longitudinal panel data reshaped from wide to long format (4 survey phases)\n"
        "- Categorical features one-hot encoded (Job type) or binary encoded (Sex, Marital Status)\n"
        "- Missing household income flagged with a dedicated `Income_Missing` indicator\n"
        "- UCLA-LS3 scores binarized at the cutoff of **22** → Low vs. High Loneliness\n"
        "- Phases 1--3 used for training (n=7,977); Phase 4 (2024) held out for testing (n=2,659)\n\n"
        "**2. Primary Model: TabPFN**\n"
        "- TabPFN is a Prior-Data Fitted Network — a "
        "meta-learned transformer that performs Bayesian inference in a single forward pass\n"
        "- It requires no hyperparameter tuning and is especially strong on tabular data with "
        "moderate sample sizes\n"
        "- The model takes **26 features** (demographics, lifestyle, COVID impact) and outputs "
        "class probabilities for Low/High loneliness\n\n"
        "**3. XGBoost Surrogate**\n"
        "- An XGBoost classifier (500 trees, depth 6, lr=0.05, subsample=0.8, "
        "colsample_bytree=0.8) is trained to mimic TabPFN's predictions on the "
        "training set, with sample weights emphasising high-confidence TabPFN "
        "predictions\n"
        "- XGBoost was selected by a notebook benchmark that compared LightGBM, "
        "XGBoost, Random Forest, Decision Tree, and Logistic Regression on test-set "
        "agreement and Spearman rank correlation against TabPFN's probabilities — "
        "XGBoost won on test agreement\n"
        "- The surrogate enables **exact TreeSHAP** computation (polynomial time), "
        "which would not be feasible directly on TabPFN\n"
        "- The surrogate also powers the **What-If Analysis** for instant "
        "interactive predictions"
    )

    st.subheader("Why SHAP and DiCE?")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)**\n\n"
        "SHAP was selected because it produces **both global and local feature importance** "
        "from the same set of values. Averaging the absolute SHAP values across the test set "
        "gives population-level importance (which features matter overall), while the "
        "per-sample SHAP vector explains an individual prediction (why *this* person was "
        "classified the way they were). The values are grounded in cooperative game theory — "
        "they are the unique attribution that satisfies efficiency, symmetry, dummy, and "
        "additivity, so feature contributions sum exactly to the model output minus its "
        "expected value. We compute **TreeSHAP** on the XGBoost surrogate, which gives "
        "*exact* SHAP values in polynomial time — substantially faster than KernelSHAP "
        "(which is model-agnostic but approximate and orders of magnitude slower on this "
        "dataset).\n\n"
        "**DiCE (Diverse Counterfactual Explanations)**\n\n"
        "DiCE generates **diverse, actionable** counterfactuals — multiple alternative "
        "scenarios that would flip the model's prediction, while differing from each other "
        "(diversity) and staying close to the original instance (proximity). Crucially, "
        "DiCE supports first-class constraints via `features_to_vary` and `permitted_range`, "
        "which let us *forbid* changes to immutable attributes such as Age and Sex. This is "
        "what makes the **Demographics-Excluded** counterfactual set in the explorer below "
        "meaningful: a recommendation a person can actually act on, not one that requires "
        "becoming younger or changing biological sex. DiCE is model-agnostic, so the same "
        "explainer wraps TabPFN directly — no surrogate needed for counterfactual search."
    )
    st.markdown("---")

    if precomputed_preds is not None:
        y_pred = precomputed_preds["y_pred"]
        y_prob = precomputed_preds["y_prob"]
    else:
        with st.spinner("Running model predictions..."):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("How well does the model perform?")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", f"{auc:.4f}",
                help="How well the model distinguishes between classes. "
                     "1.0 = perfect separation, 0.5 = random guessing.")
    col2.metric("F1-Score", f"{f1:.4f}",
                help="Balance between precision and recall. Ranges 0 to 1.")
    col3.metric("Accuracy", f"{acc:.1%}",
                help="Percentage of correct predictions out of all predictions made.")

    st.markdown(
        f"The model achieves an **ROC AUC of {auc:.4f}**, meaning it has a "
        f"**{auc:.1%}** chance of correctly ranking a randomly chosen High Loneliness "
        f"individual above a randomly chosen Low Loneliness individual. "
        f"It was evaluated on {len(y_test)} individuals from the 2024 survey wave."
    )

    st.subheader("Confusion Matrix")
    st.markdown(
        "This table shows how many predictions were correct vs. incorrect. "
        "Rows are actual outcomes; columns are what the model predicted."
    )
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=[f"Predicted {n}" for n in class_names],
        y=[f"Actual {n}" for n in class_names],
        annotation_text=[[str(y) for y in x] for x in cm.tolist()],
        colorscale="Blues"
    )
    fig_cm.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig_cm, width="stretch")

    st.subheader("Prediction Confidence Distribution")
    st.markdown(
        "This histogram shows how confident the model is in its predictions. "
        "Values near 0 mean confident about Low Loneliness; "
        "values near 1 mean confident about High Loneliness."
    )
    prob_df = pd.DataFrame({"Probability (High Loneliness)": y_prob})
    fig = px.histogram(prob_df, x="Probability (High Loneliness)", nbins=30,
                       color_discrete_sequence=["steelblue"])
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Decision boundary")
    st.plotly_chart(fig, width="stretch")
