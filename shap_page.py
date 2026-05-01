import streamlit as st
import numpy as np

from model import (
    _import_shap, _import_matplotlib, get_display_name, format_feature_value,
    build_person_options, get_prediction, build_profile_df,
)

# Template Explanations
# The initial idea was to connect this to a LLM to generate dynamic explanations.
# That costs moneyyyy so we'll go with templates instead :ok:
def _explain_waterfall(sample_values, shap_vals, features, display_names,
                       category_labels, likert_features, class_names, pred_class):
    ranked = sorted(zip(features, shap_vals, [sample_values[f] for f in features]),
                    key=lambda x: abs(x[1]), reverse=True)
    toward = [(f, s, v) for f, s, v in ranked if s > 0][:3]
    against = [(f, s, v) for f, s, v in ranked if s < 0][:3]

    def _fmt(items, sign):
        return "\n".join(
            f"- **{get_display_name(f, display_names)}** = "
            f"{format_feature_value(f, v, category_labels, likert_features)} "
            f"(contribution: {'+' if sign else ''}{s:.3f})"
            for f, s, v in items
        )

    lines = [f"**Why was this person classified as {class_names[pred_class]}?**\n"]
    if toward:
        lines += ["The main factors **pushing toward High Loneliness** are:", _fmt(toward, True)]
    if against:
        lines += ["\nThe main factors **pushing toward Low Loneliness** are:", _fmt(against, False)]
    lines.append("\nEach factor's contribution shows how much it pushes the prediction "
                 "away from the average. Larger values mean stronger influence.")
    return "\n".join(lines)

def _explain_global(shap_vals_class, features, display_names, class_name):
    ranked = sorted(zip(features, np.abs(shap_vals_class).mean(axis=0)),
                    key=lambda x: x[1], reverse=True)
    top5 = "\n".join(f"- **{get_display_name(f, display_names)}** (avg impact: {v:.4f})"
                     for f, v in ranked[:5])
    return (f"**What drives the model's {class_name} predictions?**\n\n"
            f"The top factors are:\n{top5}\n\n"
            "Features at the top have the strongest influence. "
            "See the *Feature Effects* tab for direction.")

# Page Renderer
def render_shap(model, X_explain, shap_values_test, shap_expected_value,
                features, class_names, display_names, category_labels,
                likert_features, precomputed_preds):
    shap = _import_shap()
    plt = _import_matplotlib()

    st.title("🔍 SHAP Explanations")
    st.info(
        "**SHAP (SHapley Additive exPlanations)** shows how each feature contributes "
        "to the model's prediction for each person. "
        "SHAP values are computed via an **XGBoost surrogate** trained to mimic the "
        "primary TabPFN model, using exact TreeSHAP."
    )

    n_explained, n_features = len(X_explain), len(features)
    disp_names = [get_display_name(f, display_names) for f in features]
    fig_h = max(5, n_features * 0.28)

    # Both Global Importance and Feature Effects focus on the High Loneliness
    # medyo redundant kung both low and high ipapakita,,,
    high_idx = 1
    high_name = class_names[high_idx]

    tab1, tab2, tab3 = st.tabs([
        "🌐 Global Feature Importance", "📈 Feature Effects", "👤 Individual Explanation"
    ])

    # Tab 1: Global Feature Importance
    with tab1:
        st.subheader(f"Which features matter most for predicting {high_name}?")
        st.markdown(
            f"Average importance across **{n_explained}** individuals. "
            f"All **{n_features}** features shown for the **{high_name}** class. "
            "The **top 5 most influential features are highlighted in red**."
        )

        mean_abs = np.abs(shap_values_test[high_idx]).mean(axis=0)
        order = np.argsort(mean_abs)
        names_s = [disp_names[i] for i in order]
        vals_s = mean_abs[order]
        # Top 5 = the 5 largest mean(|SHAP|) values
        n_top = 5
        colors = (["steelblue"] * (len(vals_s) - n_top)) + (["#d62728"] * n_top)

        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.barh(names_s, vals_s, color=colors)
        ax.set_xlabel("Mean(|SHAP|) value — impact on High Loneliness prediction")
        ax.set_title(f"Global Feature Importance ({high_name})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("📖 What does this mean?", expanded=True):
            st.markdown(_explain_global(
                shap_values_test[high_idx], features, display_names, high_name))

        with st.expander("📐 Understanding SHAP Impact Values in Classification"):
            st.markdown(
                "SHAP values represent contributions to the **log-odds**.\n\n"
                "| Mean(\\|SHAP\\|) | Interpretation |\n|---|---|\n"
                "| **≥ 1.5** | Very strong — multiplies odds by ~4.5× |\n"
                "| **1.0–1.5** | Strong |\n| **0.5–1.0** | Moderate |\n"
                "| **< 0.5** | Mild to weak |\n\n"
                "A feature with impact **2.0** multiplies odds by ~**7.4×** vs ~**2.7×** "
                "for impact **1.0** — the effect is exponential, not linear. "
                "🔴 Red bars in the chart mark the top 5 most influential features."
            )

    # Tab 2: Feature Effects
    with tab2:
        st.subheader(f"How do feature values push predictions toward {high_name}?")
        st.markdown(
            "Each dot is one person. "
            "**Red dots** = high feature values, **blue** = low. "
            "Dots on the right push toward **High Loneliness**; dots on "
            "the left push toward Low Loneliness."
        )

        fig, _ = plt.subplots(figsize=(8, fig_h))
        shap.summary_plot(shap_values_test[high_idx], X_explain,
                          feature_names=disp_names, show=False,
                          max_display=n_features)
        plt.title(f"Feature Effects on {high_name} prediction")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("📖 How to read this chart"):
            st.markdown(
                "- Each dot = one person\n"
                "- **Horizontal**: how much the feature pushed the prediction "
                "toward High Loneliness (right) or Low Loneliness (left)\n"
                "- **Color**: feature value (red=high, blue=low)\n"
                "- Red dots on the right → high values push toward High Loneliness\n"
                "- Sorted by importance (top = most important)"
            )

    # Tab 3: Individual Explanation
    with tab3:
        st.subheader("Why was this specific person classified this way?")
        st.markdown("Red bars push toward High Loneliness; blue toward Low.")

        options, labels = build_person_options(X_explain, class_names, precomputed_preds)
        sample_idx = st.selectbox("Select a person:", options,
                                  format_func=lambda x: labels[x], key="shap_person")

        pred, prob = get_prediction(model, X_explain, sample_idx, precomputed_preds)

        c1, c2 = st.columns(2)
        c1.metric("Predicted Class", class_names[pred])
        c2.metric("Confidence", f"{prob[pred]:.1%} {class_names[pred]}")

        st.markdown("**Person's Profile**")
        st.dataframe(build_profile_df(X_explain.iloc[sample_idx], features,
                                      display_names, category_labels, likert_features,
                                      include_raw=True),
                     width="stretch", hide_index=True)
        st.markdown(f"**Predicted class:** {class_names[pred]} ({prob[pred]:.1%})")
        st.markdown(
            "**How to read the numbers:** *Raw* = numeric dataset value. "
            "Likert features range 1–7. Binary features map 0/1 to *Value* labels. "
            "Age is in years; Income is ordinal."
        )

        base_val = (shap_expected_value[1] if len(shap_expected_value) > 1
                    else float(shap_expected_value[0]))
        explanation = shap.Explanation(
            values=shap_values_test[1][sample_idx], base_values=base_val,
            data=X_explain.iloc[sample_idx].values, feature_names=disp_names
        )

        fig, _ = plt.subplots(figsize=(10, max(7, n_features * 0.3)))
        shap.waterfall_plot(explanation, show=False, max_display=n_features)
        ax = plt.gca()
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        # Place the directional legend at the TOP of the plot (above the
        # title) so the figure does not waste vertical space on an empty
        # margin under the x-axis.
        plt.suptitle("← Low Loneliness | High Loneliness →",
                     fontsize=10, color="gray", y=1.0)
        plt.title(f"What influenced this prediction? (Person {sample_idx + 1})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("📖 What does this mean?", expanded=True):
            st.markdown(_explain_waterfall(
                X_explain.iloc[sample_idx].to_dict(),
                shap_values_test[1][sample_idx],
                features, display_names, category_labels, likert_features,
                class_names, pred
            ))