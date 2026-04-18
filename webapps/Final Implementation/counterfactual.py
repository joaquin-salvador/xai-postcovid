import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_shap, _import_matplotlib, _import_plotly,
    get_display_name, format_feature_value,
    build_person_options, get_prediction, build_profile_df,
    BINARY_FEATURES, JOB_FEATURES, JOB_OPTIONS,
)

# Feature grouping for What-If
DEMOGRAPHIC_FEATURES = ["Sex_Male",
                        "Age",
                        "Is_Married",
                        "Has_Child"]
LIFESTYLE_FEATURES = ["Activity",
                      "Exercise",
                      "Healthy_Diet",
                      "Healthy_Sleep",
                      "Interaction_Offline",
                      "Interaction_Online"]
COVID_FEATURES = ["Altruistic",
                  "Frustration",
                  "Optimism",
                  "Covid_Anxiety",
                  "Covid_Sleepless",
                  "Deterioration_Economy",
                  "Deterioration_Interact",
                  "Difficulty_Living",
                  "Difficulty_Work"]

# Helpers
def _change_direction(feat, cf_val, orig_val):
    """Return direction label: Change for categoricals, Increase/Decrease for others."""
    if feat in BINARY_FEATURES or feat in JOB_FEATURES:
        return "Change"
    return "Increase" if cf_val > orig_val else "Decrease"


def _style_direction(val):
    """Cell style for direction column."""
    if val == "Increase":
        return "background-color: #c6efce; color: #006100"
    elif val == "Decrease":
        return "background-color: #ffc7ce; color: #9c0006"
    return "background-color: #fff2cc; color: #7f6003"


def _render_styled_changes(changes_df):
    """Render a styled dataframe with colored Direction column."""
    styler = changes_df.style
    apply = getattr(styler, "map", None) or styler.applymap
    st.dataframe(apply(_style_direction, subset=["Direction"]),
                 width="stretch", hide_index=True)


def _render_feature_control(feat, orig_val, display_names, category_labels,
                            likert_features, feat_min, feat_max, key_prefix="wi"):
    """Render a single feature control widget and return the modified value."""
    dname = get_display_name(feat, display_names)

    if feat in BINARY_FEATURES:
        if feat in category_labels:
            opts = category_labels[feat]
            return float(st.selectbox(dname, list(opts.keys()),
                                      format_func=lambda x, m=opts: m[x],
                                      index=int(orig_val), key=f"{key_prefix}_{feat}"))
        return float(st.selectbox(dname, [0, 1], index=int(orig_val),
                                  key=f"{key_prefix}_{feat}"))
    elif feat == "Income":
        cats = category_labels.get("Income", {})
        return float(st.selectbox(dname, list(cats.keys()),
                                  format_func=lambda x, m=cats: m[x],
                                  index=int(orig_val), key=f"{key_prefix}_{feat}"))
    elif feat in likert_features:
        return float(st.slider(f"{dname} (1-7)", 1, 7,
                               int(round(orig_val)), key=f"{key_prefix}_{feat}"))
    elif feat == "Age":
        return float(st.slider(dname, 19, 92, int(round(orig_val)),
                               key=f"{key_prefix}_{feat}"))
    else:
        lo = min(int(feat_min[feat]), int(round(orig_val)))
        hi = max(int(feat_max[feat]), int(round(orig_val)))
        return float(st.slider(dname, lo, hi, int(round(orig_val)),
                               key=f"{key_prefix}_{feat}"))


def _render_feature_group(title, icon, caption, feat_list, features, original,
                          display_names, category_labels, likert_features,
                          feat_min, feat_max, modified_values, image=None):
    """Render a group of feature controls with a header."""
    st.markdown(f"#### {icon} {title}")
    st.caption(caption)
    if image:
        img_l, img_c, img_r = st.columns([1, 2, 1])
        with img_c:
            st.image(image, use_container_width=True)
    cols = st.columns(2)
    col_idx = 0
    for feat in feat_list:
        if feat not in features:
            continue
        with cols[col_idx % 2]:
            modified_values[feat] = _render_feature_control(
                feat, float(original[feat]), display_names, category_labels,
                likert_features, feat_min, feat_max)
        col_idx += 1

# Template Explanations
def _explain_whatif(original_values, modified_values, features, display_names,
                    category_labels, likert_features, shap_vals,
                    class_names, orig_pred, mod_pred, orig_prob, mod_prob):
    changed = [(feat, float(original_values[feat]), float(modified_values[feat]),
                shap_vals[features.index(feat)])
               for feat in features
               if abs(float(modified_values[feat]) - float(original_values[feat])) > 1e-9]
    if not changed:
        return "No features were changed. Adjust the sliders to see how the prediction changes."

    changed.sort(key=lambda x: abs(x[3]), reverse=True)
    flipped = orig_pred != mod_pred

    if flipped:
        header = (f"The prediction **changed** from **{class_names[orig_pred]}** "
                  f"to **{class_names[mod_pred]}** "
                  f"(probability: {orig_prob[1]:.1%} -> {mod_prob[1]:.1%}).")
    else:
        header = (f"The prediction **stayed** as **{class_names[orig_pred]}** "
                  f"(probability: {orig_prob[1]:.1%} -> {mod_prob[1]:.1%}).")

    details = []
    for feat, orig, mod, sv in changed[:5]:
        dname = get_display_name(feat, display_names)
        importance = "high" if abs(sv) > 0.05 else "moderate" if abs(sv) > 0.02 else "low"
        direction = "toward High Loneliness" if sv > 0 else "toward Low Loneliness"
        details.append(
            f"- **{dname}**: {format_feature_value(feat, orig, category_labels, likert_features)} "
            f"-> {format_feature_value(feat, mod, category_labels, likert_features)} "
            f"({importance} importance, originally pushed {direction})")

    footer = ("The combination of changes tipped the prediction." if flipped
              else "Try adjusting features with higher importance values.")
    return f"{header}\n\n**Changes:**\n" + "\n".join(details) + f"\n\n{footer}"


def _explain_counterfactual(original_values, cf, features, display_names,
                            category_labels, likert_features, class_names, original_pred):
    target = class_names[1 - original_pred]
    changes = [(f, original_values[f], cf[f])
               for f in features
               if f in cf and f in original_values and abs(cf[f] - original_values[f]) > 0.01]
    if not changes:
        return "No significant changes were needed in this counterfactual scenario."

    items = "\n".join(
        f"- **{get_display_name(f, display_names)}**: "
        f"{format_feature_value(f, o, category_labels, likert_features)} -> "
        f"{format_feature_value(f, c, category_labels, likert_features)} "
        f"({_change_direction(f, c, o).lower()})"
        for f, o, c in changes)
    return (f"To change the prediction to **{target}**, these changes would be needed:\n\n"
            f"{items}\n\nThese are hypothetical scenarios, not guaranteed real-world outcomes.")

# What-Ifs
def render_whatif(model, surrogate, X_train, X_explain, shap_values_test,
                  shap_expected_value, features, class_names,
                  display_names, category_labels, likert_features,
                  precomputed_preds):
    shap = _import_shap()
    plt = _import_matplotlib()
    predict_model = surrogate if surrogate is not None else model

    st.title("🔧 What-If Analysis")
    st.info(
        "Explore how changing a person's characteristics would affect the prediction."
        + (" Predictions use the **LightGBM surrogate** for fast response."
           if surrogate else ""))

    options, labels = build_person_options(X_explain, class_names, precomputed_preds)
    sample_idx = st.selectbox("Start from person:", options,
                              format_func=lambda x: labels[x], key="wi_sample")

    original = X_explain.iloc[sample_idx]
    orig_pred, orig_prob = get_prediction(model, X_explain, sample_idx, precomputed_preds)

    feat_min = X_train[features].min()
    feat_max = X_train[features].max()
    modified_values = {}

    controls_col, results_col = st.columns([1, 1])

    with controls_col:
        st.markdown("---")
        st.subheader("Adjust Features")

        _render_feature_group("Demographics", "👤",
                              "Sex, Age, Marital Status, Children, Income, Occupation",
                              DEMOGRAPHIC_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/demographics.png")

        # Income + Occupation side-by-side
        inc_col, occ_col = st.columns(2)
        with inc_col:
            if "Income" in features:
                modified_values["Income"] = _render_feature_control(
                    "Income", float(original["Income"]), display_names,
                    category_labels, likert_features, feat_min, feat_max)
        with occ_col:
            orig_job = next((jf for jf in JOB_FEATURES
                             if jf in features and float(original[jf]) == 1.0), "Job_Employed")
            selected_job = st.selectbox("Occupation", JOB_FEATURES,
                                        format_func=lambda x: JOB_OPTIONS[x],
                                        index=JOB_FEATURES.index(orig_job), key="wi_occupation")
            for jf in JOB_FEATURES:
                modified_values[jf] = 1.0 if jf == selected_job else 0.0

        # Auto-derive Income_Missing from Income selection
        if "Income_Missing" in features:
            inc_val = modified_values.get("Income")
            inc_labels = category_labels.get("Income", {})
            inc_label = inc_labels.get(int(inc_val), "") if inc_val is not None else ""
            modified_values["Income_Missing"] = (
                1.0 if "missing" in str(inc_label).lower() else 0.0)

        st.markdown("---")
        _render_feature_group("Lifestyle", "🏃",
                              "Activity, Exercise, Diet, Sleep, Social Interaction",
                              LIFESTYLE_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/lifestyle.png")

        st.markdown("---")
        _render_feature_group("COVID Impact", "🦠",
                              "Anxiety, Sleep, Economic & Social Deterioration, Difficulties",
                              COVID_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/covid.png")

        # Any remaining features not in the groups
        remaining = [f for f in features if f not in modified_values and f not in JOB_FEATURES]
        if remaining:
            st.markdown("---")
            _render_feature_group("Other Features", "📋", "",
                                  remaining, features, original,
                                  display_names, category_labels, likert_features,
                                  feat_min, feat_max, modified_values)

    # Results
    modified_df = pd.DataFrame([modified_values], columns=features)
    mod_pred = predict_model.predict(modified_df)[0]
    mod_prob = predict_model.predict_proba(modified_df)[0]
    changed_feats = [f for f in features
                     if abs(modified_values[f] - float(original[f])) > 1e-9]

    with results_col:
        st.markdown("---")
        st.subheader("Results")

        left, right = st.columns(2)
        with left:
            st.markdown("##### Original")
            st.metric("Prediction", class_names[orig_pred])
            st.metric("P(High Loneliness)", f"{orig_prob[1]:.1%}")
        with right:
            st.markdown("##### Modified")
            flipped = mod_pred != orig_pred
            st.metric("Prediction", class_names[mod_pred],
                      delta="Flipped!" if flipped else "No change",
                      delta_color="normal" if flipped else "off")
            st.metric("P(High Loneliness)", f"{mod_prob[1]:.1%}",
                      delta=f"{mod_prob[1] - orig_prob[1]:+.1%}")

        if changed_feats:
            st.markdown("**What you changed:**")
            st.dataframe(pd.DataFrame([{
                "Feature": get_display_name(f, display_names),
                "Original": format_feature_value(f, float(original[f]),
                                                 category_labels, likert_features),
                "Modified": format_feature_value(f, modified_values[f],
                                                 category_labels, likert_features),
            } for f in changed_feats]), width="stretch", hide_index=True)
        else:
            st.info("No features changed yet. Adjust the controls on the left.")

        # SHAP waterfall
        n_features = len(features)
        disp_names = [get_display_name(f, display_names) for f in features]
        st.markdown("##### 📊 Original SHAP Breakdown")
        base_val = (shap_expected_value[1] if len(shap_expected_value) > 1
                    else float(shap_expected_value[0]))
        explanation = shap.Explanation(
            values=shap_values_test[1][sample_idx], base_values=base_val,
            data=original.values, feature_names=disp_names)
        fig, _ = plt.subplots(figsize=(8, max(6, n_features * 0.25)))
        shap.waterfall_plot(explanation, show=False, max_display=n_features)
        plt.title("Original Prediction Breakdown")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if changed_feats:
            with st.expander("📖 What does this mean?", expanded=True):
                st.markdown(_explain_whatif(
                    original.to_dict(), modified_values, features, display_names,
                    category_labels, likert_features, shap_values_test[1][sample_idx],
                    class_names, orig_pred, mod_pred, orig_prob, mod_prob))

    # Scale descriptions
    st.markdown("---")
    with st.expander("📏 Feature Scale Descriptions"):
        st.markdown(
            "**Demographics**\n\n"
            "| Feature | Scale |\n|---|---|\n"
            "| Sex | Binary: Male / Female |\n| Age | Years (19-92) |\n"
            "| Marital Status | Binary: Married / Unmarried |\n"
            "| Has Children | Binary: Yes / No |\n"
            "| Household Income | Ordinal categories (JPY brackets, incl. Missing) |\n"
            "| Occupation | One-hot: Employed, Homemaker, Student, Unemployed, Other |\n\n"
            "**Lifestyle** _(all 1-7 Likert scale)_\n\n"
            "| Feature | Description |\n|---|---|\n"
            "| Social/Physical Activity | _To be added_ |\n"
            "| Exercise Frequency | _To be added_ |\n"
            "| Healthy Diet | _To be added_ |\n"
            "| Healthy Sleep | _To be added_ |\n"
            "| Offline Social Interaction | _To be added_ |\n"
            "| Online Social Interaction | _To be added_ |\n\n"
            "**COVID Impact** _(all 1-7 Likert scale)_\n\n"
            "| Feature | Description |\n|---|---|\n"
            "| Altruistic Behavior | _To be added_ |\n"
            "| Frustration Level | _To be added_ |\n"
            "| Optimism Level | _To be added_ |\n"
            "| COVID Anxiety | _To be added_ |\n"
            "| COVID-related Sleeplessness | _To be added_ |\n"
            "| Economic Deterioration | _To be added_ |\n"
            "| Social Interaction Deterioration | _To be added_ |\n"
            "| Difficulty Living | _To be added_ |\n"
            "| Difficulty Working | _To be added_ |"
        )

# Counterfactual Explorer
def render_counterfactuals(model, X_test, y_test, features, class_names,
                           feature_info, display_names, category_labels,
                           likert_features, precomputed_cfs, precomputed_preds):
    st.title("🔄 Counterfactual Explorer")
    st.info(
        "Counterfactual explanations answer: **\"What would need to change for this "
        "person's prediction to be different?\"** They show the smallest changes needed "
        "to flip the model's decision."
    )

    if precomputed_cfs is not None:
        available_indices = sorted(precomputed_cfs.keys())
        st.success(f"Loaded **{len(available_indices)}** pre-computed counterfactual scenarios.")
    else:
        available_indices = list(range(len(X_test)))
        st.warning("No pre-computed counterfactuals found. Run the notebook first.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Person")
        options, labels = build_person_options(
            X_test.iloc[available_indices] if precomputed_cfs else X_test,
            class_names, precomputed_preds)
        # Map back to original indices
        idx_map = {i: available_indices[i] for i in range(len(available_indices))} if precomputed_cfs else None
        sample_pos = st.selectbox("Select a person:", available_indices,
                                  format_func=lambda x: labels.get(
                                      available_indices.index(x) if precomputed_cfs else x,
                                      f"Person {x+1}"),
                                  key="cf_sample")
        sample_idx = sample_pos
        instance = X_test.iloc[[sample_idx]]

        pred, prob = get_prediction(model, X_test, sample_idx, precomputed_preds)
        if pred is not None:
            st.markdown(f"**Current Prediction**: {class_names[pred]} "
                        f"({prob[pred]:.1%} confidence)")

        st.markdown("**Person's Profile**")
        st.dataframe(build_profile_df(instance.iloc[0], features, display_names,
                                      category_labels, likert_features),
                     width="stretch", hide_index=True)

    with col2:
        st.subheader("Counterfactual Scenarios")
        original_values = instance.iloc[0]
        cf_records = (precomputed_cfs.get(sample_idx) if precomputed_cfs else None)

        if cf_records:
            st.markdown(f"Found **{len(cf_records)}** alternative scenario(s).")
            st.markdown("**What would need to change?**")

            n_scenarios = len(cf_records)
            if n_scenarios > 1:
                nav_l, nav_c, nav_r = st.columns([1, 3, 1])
                with nav_l:
                    if st.button("◀ Prev", key="cf_prev", use_container_width=True):
                        st.session_state["cf_scenario_idx"] = (
                            st.session_state.get("cf_scenario_idx", 0) - 1) % n_scenarios
                with nav_r:
                    if st.button("Next ▶", key="cf_next", use_container_width=True):
                        st.session_state["cf_scenario_idx"] = (
                            st.session_state.get("cf_scenario_idx", 0) + 1) % n_scenarios
                si = st.session_state.get("cf_scenario_idx", 0)
                with nav_c:
                    st.markdown(f"<div style='text-align:center;padding-top:5px;'>"
                                f"<strong>Scenario {si+1} of {n_scenarios}</strong></div>",
                                unsafe_allow_html=True)
            else:
                si = 0
                st.markdown("**Scenario 1 of 1**")

            cf = cf_records[si]
            changes = [{
                "Feature": get_display_name(f, display_names),
                "Current": format_feature_value(f, original_values[f], category_labels, likert_features),
                "Needed": format_feature_value(f, cf[f], category_labels, likert_features),
                "Direction": _change_direction(f, cf[f], original_values[f]),
            } for f in features
              if f in cf and f in original_values and abs(cf[f] - original_values[f]) > 0.01]

            if changes:
                _render_styled_changes(pd.DataFrame(changes))
                st.markdown("---")
                st.markdown(_explain_counterfactual(
                    original_values.to_dict(), cf, features, display_names,
                    category_labels, likert_features, class_names,
                    pred if pred is not None else 0))
            else:
                st.info("No significant changes in this scenario.")
        else:
            st.info("No counterfactuals available for this person.")

    # Summary chart
    if cf_records and len(cf_records) > 1:
        px, _ = _import_plotly()
        st.markdown("---")
        st.subheader("📊 Summary: Most Frequently Changed Features")

        feat_counts = {}
        for cf in cf_records:
            for f in features:
                if f in cf and f in original_values and abs(cf[f] - original_values[f]) > 0.01:
                    dname = get_display_name(f, display_names)
                    feat_counts[dname] = feat_counts.get(dname, 0) + 1

        if feat_counts:
            summary_df = pd.DataFrame(
                sorted(feat_counts.items(), key=lambda x: -x[1]),
                columns=["Feature", "Times Changed"])
            fig = px.bar(summary_df, x="Times Changed", y="Feature",
                         orientation="h", color_discrete_sequence=["steelblue"])
            fig.update_layout(yaxis=dict(autorange="reversed"),
                              height=max(300, len(feat_counts) * 30))
            st.plotly_chart(fig, width="stretch")
