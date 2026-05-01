import os
import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_plotly, get_display_name, format_feature_value,
    DATA_DIR, ARTIFACTS_DIR,
)

PHASE_YEARS = {1: 2020, 2: 2021, 3: 2022, 4: 2024}
PHASE_PALETTE = {
    1: "#4361ee",
    2: "#7209b7",
    3: "#f72585",
    4: "#fb8500",
}


@st.cache_data
def _load_encoded_dataset():
    """Load the encoded dataset"""
    path = os.path.join(ARTIFACTS_DIR, "df_encoded.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _phase_label(p):
    return f"Phase {int(p)} ({PHASE_YEARS.get(int(p), '?')})"


def _centered(df, precision=3):
    """Return a Pandas Styler that centers headers + cells"""
    return (df.style
              .format(precision=precision)
              .set_properties(**{"text-align": "center"})
              .set_table_styles([
                  {"selector": "th",
                   "props": [("text-align", "center"),
                             ("white-space", "nowrap")]},
                  {"selector": "td",
                   "props": [("white-space", "nowrap")]},
              ]))


def render_eda(df_encoded, features, target, display_names,
               category_labels, likert_features, class_names):
    """Render the EDA tab."""
    px, _ = _import_plotly()

    st.title("📈 Exploratory Data Analysis")
    st.info(
        "How does each feature's distribution shift across the four "
        "longitudinal survey waves? Each phase corresponds to a year: "
        "**Phase 1 → 2020**, **Phase 2 → 2021**, **Phase 3 → 2022**, "
        "**Phase 4 → 2024**. Use the controls below to drill into any "
        "feature."
    )

    if df_encoded is None or "Phase" not in df_encoded.columns:
        st.warning(
            "`artifacts/df_encoded.csv` is not present. Run the notebook's "
            "**§7. Save & Verify Artifacts** cell to produce it, then "
            "reload this page."
        )
        return

    phases = sorted(df_encoded["Phase"].dropna().unique().astype(int).tolist())
    if not phases:
        st.warning("No phase column data found in df_encoded.csv.")
        return

    # Summary cards
    n_per_phase = int((df_encoded["Phase"] == phases[0]).sum())
    st.metric("Participants per phase", f"{n_per_phase:,}",
              help="The same panel of respondents was surveyed in each "
                   "phase, so the participant count is identical across "
                   "phases.")

    cols = st.columns(len(phases))
    for col, p in zip(cols, phases):
        rate = float(df_encoded.loc[df_encoded["Phase"] == p, target].mean())
        col.metric(_phase_label(p), f"P(High) = {rate:.1%}")

    st.markdown("---")

    tab_overview, tab_feature, tab_target = st.tabs([
        "🌐 All-feature overview",
        "🔬 Drill into one feature",
        "🎯 Target rate by phase",
    ])

    # =================================================================
    # Tab 1 — overview: per-feature mean trend across phases
    # =================================================================
    with tab_overview:
        st.subheader("Per-feature mean across phases")
        st.caption(
            "Each panel shows one feature's mean per phase. Watch for "
            "shifts that line up with COVID milestones (lockdowns, "
            "reopenings)."
        )

        mean_by_phase = (
            df_encoded.groupby("Phase")[features].mean().T
        )
        mean_by_phase.columns = [_phase_label(p) for p in mean_by_phase.columns]
        mean_by_phase.index.name = "Feature"
        mean_by_phase["Display Name"] = [
            get_display_name(f, display_names) for f in mean_by_phase.index
        ]

        long_df = mean_by_phase.reset_index().melt(
            id_vars=["Feature", "Display Name"],
            value_vars=[c for c in mean_by_phase.columns if c.startswith("Phase ")],
            var_name="Phase", value_name="Mean",
        )

        fig = px.line(
            long_df, x="Phase", y="Mean",
            facet_col="Display Name", facet_col_wrap=4,
            markers=True,
            color_discrete_sequence=["steelblue"],
            height=140 * ((len(features) + 3) // 4),
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
        fig.for_each_annotation(lambda a: a.update(
            text=a.text.split("=")[-1], font_size=10))
        fig.for_each_yaxis(lambda y: y.update(matches=None,
                                              showticklabels=True,
                                              tickfont=dict(size=8)))
        fig.for_each_xaxis(lambda x: x.update(tickfont=dict(size=7)))
        st.plotly_chart(fig, width="stretch")

        with st.expander("📋 Mean per feature per phase (table)"):
            tbl = (mean_by_phase[[c for c in mean_by_phase.columns
                                  if c.startswith("Phase ")] + ["Display Name"]]
                   .reset_index().rename(columns={"Feature": "Raw Feature"})
                   .set_index("Display Name")
                   .round(3))
            st.table(_centered(tbl))

    # =================================================================
    # Tab 2 — drill into one feature
    # =================================================================
    with tab_feature:
        st.subheader("Feature drill-down")

        feat = st.selectbox(
            "Feature to explore:",
            features,
            format_func=lambda f: get_display_name(f, display_names),
            key="eda_feat",
        )
        feat_label = get_display_name(feat, display_names)

        df_sub = df_encoded[["Phase", feat, target]].dropna()
        df_sub["Phase Label"] = df_sub["Phase"].map(_phase_label)
        df_sub["Class"] = df_sub[target].map(
            {0: class_names[0], 1: class_names[1]}
        )

        # ── Boxplot or histogram depending on feature shape ──
        unique_vals = df_sub[feat].nunique()
        if unique_vals <= 7:
            # categorical / Likert — show stacked bar of counts per phase
            counts = (
                df_sub.groupby(["Phase Label", feat]).size()
                .reset_index(name="Count")
            )
            counts[feat_label] = counts[feat].apply(
                lambda v: format_feature_value(
                    feat, v, category_labels, likert_features))
            fig = px.bar(
                counts, x="Phase Label", y="Count", color=feat_label,
                barmode="stack",
                title=f"Counts of {feat_label} per phase",
            )
        else:
            # continuous — boxplot per phase
            fig = px.box(
                df_sub, x="Phase Label", y=feat, color="Phase Label",
                points=False,
                color_discrete_map={
                    _phase_label(p): PHASE_PALETTE[p] for p in phases
                },
                title=f"Distribution of {feat_label} by phase",
            )
            fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        # Per-class median per phase
        med = (
            df_sub.groupby(["Phase", "Class"])[feat]
            .median().reset_index()
        )
        med["Phase Label"] = med["Phase"].map(_phase_label)
        fig2 = px.line(
            med, x="Phase Label", y=feat, color="Class",
            markers=True,
            color_discrete_map={
                class_names[0]: "steelblue", class_names[1]: "coral",
            },
            title=f"Median of {feat_label} by phase, split by class",
        )
        st.plotly_chart(fig2, width="stretch")

        # Summary table
        summary = df_sub.groupby("Phase")[feat].agg(
            ["count", "mean", "std", "median",
             lambda s: s.quantile(0.25),
             lambda s: s.quantile(0.75)]
        )
        summary.columns = ["count", "mean", "std", "median", "q25", "q75"]
        summary.index = [_phase_label(p) for p in summary.index]
        st.table(_centered(summary.round(3)))

    # =================================================================
    # Tab 3 — target rate by phase
    # =================================================================
    with tab_target:
        st.subheader(f"{class_names[1]} rate by phase")
        st.caption(
            "Share of respondents classified as **High Loneliness** "
            "(UCLA-LS3 ≥ 22). This is the prediction target — its "
            "phase-level shift gives context for the model task."
        )

        rate = (
            df_encoded.groupby("Phase")[target].mean()
            .rename("High Loneliness Rate").reset_index()
        )
        rate["Phase Label"] = rate["Phase"].map(_phase_label)

        fig = px.bar(
            rate, x="Phase Label", y="High Loneliness Rate",
            text_auto=".2%",
            color="Phase Label",
            color_discrete_map={
                _phase_label(p): PHASE_PALETTE[p] for p in phases
            },
        )
        fig.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig, width="stretch")

        cnt = (
            df_encoded.groupby(["Phase", target]).size().unstack(fill_value=0)
        )
        cnt.columns = [class_names[c] for c in cnt.columns]
        cnt.index = [_phase_label(p) for p in cnt.index]
        cnt["Total"] = cnt.sum(axis=1)
        st.table(_centered(cnt, precision=0))
