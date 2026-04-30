import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_shap, _import_matplotlib, _import_plotly,
    get_display_name, format_feature_value,
    build_person_options, build_person_label, get_prediction, build_profile_df,
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
                          feat_min, feat_max, modified_values, image=None,
                          key_prefix="wi"):
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
                likert_features, feat_min, feat_max, key_prefix=key_prefix)
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
        + (" Predictions use the **XGBoost surrogate** for fast response."
           if surrogate else ""))

    options, labels = build_person_options(X_explain, class_names, precomputed_preds)
    sample_idx = st.selectbox("Start from person:", options,
                              format_func=lambda x: labels[x], key="wi_sample")

    original = X_explain.iloc[sample_idx]
    orig_pred, orig_prob = get_prediction(model, X_explain, sample_idx, precomputed_preds)

    feat_min = X_train[features].min()
    feat_max = X_train[features].max()
    modified_values = {}
    key_prefix = f"wi_{sample_idx}"

    controls_col, results_col = st.columns([1, 1])

    with controls_col:
        st.markdown("---")
        st.subheader("Adjust Features")

        _render_feature_group("Demographics", "👤",
                              "Sex, Age, Marital Status, Children, Income, Occupation",
                              DEMOGRAPHIC_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/demographics.png",
                              key_prefix=key_prefix)

        # Income + Occupation side-by-side
        inc_col, occ_col = st.columns(2)
        with inc_col:
            if "Income" in features:
                modified_values["Income"] = _render_feature_control(
                    "Income", float(original["Income"]), display_names,
                    category_labels, likert_features, feat_min, feat_max,
                    key_prefix=key_prefix)
        with occ_col:
            orig_job = next((jf for jf in JOB_FEATURES
                             if jf in features and float(original[jf]) == 1.0), "Job_Employed")
            selected_job = st.selectbox("Occupation", JOB_FEATURES,
                                        format_func=lambda x: JOB_OPTIONS[x],
                                        index=JOB_FEATURES.index(orig_job),
                                        key=f"{key_prefix}_occupation")
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
                              image="assets/lifestyle.png",
                              key_prefix=key_prefix)

        st.markdown("---")
        _render_feature_group("COVID Impact", "🦠",
                              "Anxiety, Sleep, Economic & Social Deterioration, Difficulties",
                              COVID_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/covid.png",
                              key_prefix=key_prefix)

        # Any remaining features not in the groups
        remaining = [f for f in features if f not in modified_values and f not in JOB_FEATURES]
        if remaining:
            st.markdown("---")
            _render_feature_group("Other Features", "📋", "",
                                  remaining, features, original,
                                  display_names, category_labels, likert_features,
                                  feat_min, feat_max, modified_values,
                                  key_prefix=key_prefix)

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
            "Likert items are scored from **1 = *Not at all true*** to "
            "**7 = *Very true***. Each row shows the original survey "
            "question that the respondent answered.\n\n"
            "**Demographics**\n\n"
            "| Feature | Scale |\n|---|---|\n"
            "| Sex | Binary: Male / Female |\n"
            "| Age | Years (19-92) |\n"
            "| Marital Status | Binary: Married / Unmarried |\n"
            "| Has Children | Binary: Yes / No |\n"
            "| Household Income | Ordinal categories (JPY brackets, incl. Missing) |\n"
            "| Occupation | One-hot: Employed, Homemaker, Student, Unemployed, Other |\n\n"
            "**Lifestyle** _(1 = Not at all true … 7 = Very true)_\n\n"
            "| Feature | Survey item |\n|---|---|\n"
            "| Social/Physical Activity | I engaged in hobbies and other activities that I could become passionate about. |\n"
            "| Exercise Frequency | I try to exercise to stay healthy (both indoors and outdoors). |\n"
            "| Healthy Diet | I ate meals with nutritional balance in mind. |\n"
            "| Healthy Sleep | My wake-up and bedtimes were pretty consistent. |\n"
            "| Offline Social Interaction | I interacted with family and friends in person (excluding work and classes). |\n"
            "| Online Social Interaction | I interacted with family and friends online via chat or video calls (excluding work or class). |\n\n"
            "**COVID Impact** _(1 = Not at all true … 7 = Very true)_\n\n"
            "| Feature | Survey item |\n|---|---|\n"
            "| Altruistic Behavior | I voluntarily took preventive actions (mask, hand-washing, limiting going out, etc.) to prevent spreading COVID-19 to family and others. |\n"
            "| Frustration Level | Changes in my life sometimes made me irritable and angry. |\n"
            "| Optimism Level | I thought positively about the future. |\n"
            "| COVID Anxiety | Watching the news about the new coronavirus made me feel nervous and anxious. |\n"
            "| COVID-related Sleeplessness | I couldn't sleep because I was worried about catching the new coronavirus. |\n"
            "| Economic Deterioration | The economic situation worsened. |\n"
            "| Social Interaction Deterioration | Relationships with close people such as family and friends have deteriorated. |\n"
            "| Difficulty Living | Daily life was disrupted by shortages of COVID-19 prevention supplies (masks, thermometers, etc.) and other daily necessities. |\n"
            "| Difficulty Working | Changes in my lifestyle have caused problems with my work and studies. |"
        )

# Counterfactual Explorer helpers
def _scenario_changes_table(cf, original_values, features, display_names,
                            category_labels, likert_features):
    """Return a DataFrame of feature changes for a single scenario."""
    rows = [{
        "Feature": get_display_name(f, display_names),
        "Current": format_feature_value(f, original_values[f], category_labels, likert_features),
        "Needed": format_feature_value(f, cf[f], category_labels, likert_features),
        "Direction": _change_direction(f, cf[f], original_values[f]),
    } for f in features
      if f in cf and f in original_values and abs(cf[f] - original_values[f]) > 0.01]
    return pd.DataFrame(rows)


def _check_immutable(cf, original_values, immutable_features):
    """Return list of immutable features that were changed (constraint violations)."""
    if not immutable_features:
        return []
    return [f for f in immutable_features
            if f in cf and abs(cf[f] - original_values[f]) > 0.01]


def _render_cf_set(cf_records, original_values, features, display_names,
                   category_labels, likert_features, class_names,
                   original_pred, key_prefix, immutable_features=None):
    """Render scenario navigator + change table + explanation + frequency chart for one CF set."""
    if not cf_records:
        st.info("No counterfactuals available for this person in this set.")
        return

    if immutable_features:
        immutable_disp = ", ".join(get_display_name(f, display_names) for f in immutable_features)
        st.caption(f"🔒 These scenarios keep **{immutable_disp}** constant.")

    n_scenarios = len(cf_records)
    st.markdown(f"Found **{n_scenarios}** alternative scenario(s).")

    if n_scenarios > 1:
        si_key = f"{key_prefix}_scenario_idx"
        nav_l, nav_c, nav_r = st.columns([1, 3, 1])
        with nav_l:
            if st.button("◀ Prev", key=f"{key_prefix}_prev", use_container_width=True):
                st.session_state[si_key] = (
                    st.session_state.get(si_key, 0) - 1) % n_scenarios
        with nav_r:
            if st.button("Next ▶", key=f"{key_prefix}_next", use_container_width=True):
                st.session_state[si_key] = (
                    st.session_state.get(si_key, 0) + 1) % n_scenarios
        si = st.session_state.get(si_key, 0)
        with nav_c:
            st.markdown(f"<div style='text-align:center;padding-top:5px;'>"
                        f"<strong>Scenario {si+1} of {n_scenarios}</strong></div>",
                        unsafe_allow_html=True)
    else:
        si = 0
        st.markdown("**Scenario 1 of 1**")

    cf = cf_records[si]
    changes_df = _scenario_changes_table(
        cf, original_values, features, display_names, category_labels, likert_features)

    if not changes_df.empty:
        _render_styled_changes(changes_df)

        violations = _check_immutable(cf, original_values, immutable_features)
        if immutable_features:
            if violations:
                viol_disp = ", ".join(get_display_name(f, display_names) for f in violations)
                st.error(f"⚠️ Constraint violation — immutable feature(s) changed: {viol_disp}")
            else:
                imm_disp = ", ".join(get_display_name(f, display_names) for f in immutable_features)
                st.success(f"✓ Constraint satisfied: {imm_disp} held constant.")

        st.markdown("---")
        st.markdown(_explain_counterfactual(
            original_values.to_dict(), cf, features, display_names,
            category_labels, likert_features, class_names, original_pred))
    else:
        st.info("No significant changes in this scenario.")

    if n_scenarios > 1:
        _render_frequency_chart(cf_records, original_values, features,
                                display_names, key_prefix)


def _render_frequency_chart(cf_records, original_values, features,
                            display_names, key_prefix):
    """Bar chart of how often each feature is changed across all scenarios for the sample."""
    px, _ = _import_plotly()
    st.markdown("---")
    st.subheader("📊 Most Frequently Changed Features")

    feat_counts = {}
    for cf in cf_records:
        for f in features:
            if f in cf and f in original_values and abs(cf[f] - original_values[f]) > 0.01:
                dname = get_display_name(f, display_names)
                feat_counts[dname] = feat_counts.get(dname, 0) + 1

    if not feat_counts:
        st.info("No features were changed across the available scenarios.")
        return

    summary_df = pd.DataFrame(
        sorted(feat_counts.items(), key=lambda x: -x[1]),
        columns=["Feature", "Times Changed"])
    fig = px.bar(summary_df, x="Times Changed", y="Feature",
                 orientation="h", color_discrete_sequence=["steelblue"])
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      height=max(300, len(feat_counts) * 30))
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_freq_chart")


# Counterfactual Explorer
def render_counterfactuals(model, X_test, y_test, features, class_names,
                           feature_info, display_names, category_labels,
                           likert_features, precomputed_cfs,
                           precomputed_cfs_limited, precomputed_preds):
    st.title("🔄 Counterfactual Explorer")

    immutable_features = feature_info.get("immutable_features", []) or []
    immutable_disp = [get_display_name(f, display_names) for f in immutable_features]

    st.info(
        "Counterfactual explanations answer: **\"What would need to change for this "
        "person's prediction to be different?\"** They show the smallest changes needed "
        "to flip the model's decision."
    )

    has_unr = bool(precomputed_cfs)
    has_lim = bool(precomputed_cfs_limited)

    if has_unr and has_lim and immutable_features:
        st.markdown(
            "Two counterfactual sets are available:\n\n"
            "- **🔓 Unrestricted** — every feature is free to change.\n"
            f"- **🔒 Demographics-Excluded** — keeps **{', '.join(immutable_disp)}** fixed, "
            "so the suggested changes are actionable for the individual."
        )
    elif has_lim and immutable_features:
        st.markdown(
            f"Counterfactuals shown keep **{', '.join(immutable_disp)}** fixed."
        )

    available_unr = sorted(precomputed_cfs.keys()) if has_unr else []
    available_lim = sorted(precomputed_cfs_limited.keys()) if has_lim else []
    available = sorted(set(available_unr) | set(available_lim))

    if not available:
        st.warning("No pre-computed counterfactuals found. Run the notebook first.")
        return

    parts = []
    if has_unr:
        parts.append(f"**{len(available_unr)}** unrestricted")
    if has_lim:
        parts.append(f"**{len(available_lim)}** demographics-excluded")
    st.success(
        f"Loaded {' and '.join(parts)} CF scenario set(s) "
        f"covering {len(available)} unique persons."
    )

    sample_idx = st.selectbox(
        "Select a person:",
        available,
        format_func=lambda x: build_person_label(
            x, X_test.iloc[x], class_names, precomputed_preds),
        key="cf_sample",
    )
    instance = X_test.iloc[[sample_idx]]
    original_values = instance.iloc[0]

    pred, prob = get_prediction(model, X_test, sample_idx, precomputed_preds)
    original_pred = pred if pred is not None else 0

    profile_col, meta_col = st.columns([2, 1])
    with profile_col:
        st.subheader("Selected Person")
        if pred is not None:
            st.markdown(f"**Current Prediction**: {class_names[pred]} "
                        f"({prob[pred]:.1%} confidence)")
        st.dataframe(
            build_profile_df(original_values, features, display_names,
                             category_labels, likert_features),
            width="stretch", hide_index=True,
        )
    with meta_col:
        if immutable_features:
            st.markdown("**🔒 Immutable Features**")
            st.caption("Held constant in the demographics-excluded set.")
            for f in immutable_features:
                st.markdown(
                    f"- **{get_display_name(f, display_names)}**: "
                    f"{format_feature_value(f, float(original_values[f]), category_labels, likert_features)}"
                )
        st.markdown("**CF availability for this person**")
        if has_unr:
            mark = "✅" if sample_idx in available_unr else "—"
            st.markdown(f"- {mark} Unrestricted")
        if has_lim:
            mark = "✅" if sample_idx in available_lim else "—"
            st.markdown(f"- {mark} Demographics-Excluded")

    st.markdown("---")

    tab_labels = []
    if has_unr:
        tab_labels.append("🔓 Unrestricted")
    if has_lim:
        tab_labels.append("🔒 Demographics-Excluded")

    tabs = st.tabs(tab_labels)
    ti = 0

    if has_unr:
        with tabs[ti]:
            _render_cf_set(
                precomputed_cfs.get(sample_idx),
                original_values, features, display_names,
                category_labels, likert_features, class_names,
                original_pred, key_prefix="unr",
                immutable_features=None,
            )
        ti += 1

    if has_lim:
        with tabs[ti]:
            _render_cf_set(
                precomputed_cfs_limited.get(sample_idx),
                original_values, features, display_names,
                category_labels, likert_features, class_names,
                original_pred, key_prefix="lim",
                immutable_features=immutable_features,
            )
        ti += 1
