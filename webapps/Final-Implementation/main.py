import streamlit as st
from model import load_artifacts, DEFAULT_CATEGORY_LABELS, DEFAULT_DISPLAY_NAMES, DEFAULT_LIKERT_FEATURES

st.set_page_config(
    page_title="XAI Dashboard — UCLA Loneliness",
    page_icon="assets/computer.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
/* Colored metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea11, #764ba211);
    border: 1px solid #667eea33;
    border-radius: 10px;
    padding: 15px;
}

/* Sidebar styling — light background with legible dark text */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef2f7, #dce4ef);
}
[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}
[data-testid="stSidebar"] .stMarkdown a {
    color: #4361ee !important;
}

/* Table text wrapping — prevent truncation */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    max-width: 300px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #667eea;
}

/* Info box accent */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

def main():
    try:
        (model, surrogate, X_train, X_test, X_explain, y_train, y_test,
         shap_values_test, shap_expected_value, feature_info,
         precomputed_preds, precomputed_cfs) = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Could not load model artifacts. Please run the notebook first "
            "to generate them, then place the output files in the `data/` folder.\n\n"
            f"Error: {e}"
        )
        st.markdown(
            "**Expected folder structure:**\n"
            "```\n"
            "finalfinal_webapp/\n"
            "  data/\n"
            "    models/tabpfn.joblib\n"
            "    models/surrogate_lgbm.joblib\n"
            "    explainers/shap_values_test.pkl\n"
            "    explainers/shap_expected_value.pkl\n"
            "    explainers/feature_info.pkl\n"
            "    explainers/counterfactual_results.pkl\n"
            "    artifacts/X_train.csv\n"
            "    artifacts/X_test.csv\n"
            "    artifacts/y_train.csv\n"
            "    artifacts/y_test.csv\n"
            "    artifacts/X_explain.csv\n"
            "    artifacts/test_predictions.pkl\n"
            "```"
        )
        st.stop()

    features = feature_info["features"]
    class_names = feature_info["class_names"]
    category_labels = feature_info.get("category_labels", DEFAULT_CATEGORY_LABELS)
    display_names = feature_info.get("display_names", DEFAULT_DISPLAY_NAMES)
    likert_features = feature_info.get("likert_features", DEFAULT_LIKERT_FEATURES)

    # Sidebar
    st.sidebar.image(
        "assets/computer.png"
    )
    st.sidebar.title("UCLA Loneliness XAI Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate to:", [
        "📊 Model Overview",
        "🔍 SHAP Explanations",
        "🔧 What-If Analysis",
        "🔄 Counterfactual Explorer",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 About the Model**")
    st.sidebar.markdown(f"- Primary model: `TabPFN`")
    if surrogate is not None:
        st.sidebar.markdown(f"- Surrogate: `{type(surrogate).__name__}` (TreeSHAP + What-If)")
    st.sidebar.markdown(f"- Features: **{len(features)}**")
    st.sidebar.markdown(f"- Test samples: **{len(X_test)}**")
    st.sidebar.markdown(f"- SHAP explained: **{len(X_explain)}** samples")
    if precomputed_cfs:
        st.sidebar.markdown(f"- Counterfactuals: **{len(precomputed_cfs)}** samples")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📏 UCLA Loneliness Scale**")
    st.sidebar.markdown("🟢 Score < 22: **Low Loneliness**")
    st.sidebar.markdown("🔴 Score ≥ 22: **High Loneliness**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("All images sourced from [irasutoya.com](https://www.irasutoya.com/)")

    # Navigation
    if page == "📊 Model Overview":
        from model import render_overview
        render_overview(model, X_test, y_test, class_names, precomputed_preds)
    elif page == "🔍 SHAP Explanations":
        from shap_page import render_shap
        render_shap(model, X_explain, shap_values_test, shap_expected_value,
                    features, class_names, display_names, category_labels,
                    likert_features, precomputed_preds)
    elif page == "🔧 What-If Analysis":
        from counterfactual import render_whatif
        render_whatif(model, surrogate, X_train, X_explain, shap_values_test,
                      shap_expected_value, features, class_names,
                      display_names, category_labels, likert_features,
                      precomputed_preds)
    elif page == "🔄 Counterfactual Explorer":
        from counterfactual import render_counterfactuals
        render_counterfactuals(model, X_test, y_test, features, class_names,
                               feature_info, display_names, category_labels,
                               likert_features, precomputed_cfs, precomputed_preds)

if __name__ == "__main__":
    main()