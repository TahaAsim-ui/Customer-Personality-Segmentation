from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── Config ────────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_local = _here / "Customer_Personality_Segmentation.csv"
_sibling = _here.parent / "CustomerSegmentationPersonality" / "Customer_Personality_Segmentation.csv"
DATA_PATH = _local if _local.exists() else _sibling

RANDOM_STATE = 0
N_CLUSTERS = 3
K_ELBOW_RANGE = range(2, 11)
K_SIL_RANGE = range(2, 10)

DROP_COLS = [
    "Dt_Customer", "Year_Birth", "ID",
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    "Z_CostContact", "Z_Revenue", "Education", "Marital_Status",
]
SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

SEGMENT_COLORS = {
    "High-Value Loyalists": "#636EFA",
    "Digital Deal Seekers": "#EF553B",
    "Budget-Conscious Families": "#00CC96",
}

st.set_page_config(
    page_title="Customer Segmentation V2",
    page_icon=":busts_in_silhouette:",
    layout="wide",
)


# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    raw = pd.read_csv(DATA_PATH, sep="\t")
    n_raw = len(raw)
    n_missing = int(raw["Income"].isna().sum())
    df = raw.dropna(subset=["Income"]).copy()
    df["Age"] = date.today().year - df["Year_Birth"]
    df_model = df.drop(columns=DROP_COLS)
    return df.reset_index(drop=True), df_model.reset_index(drop=True), n_raw, n_missing


@st.cache_data
def build_pipeline(df_model: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df_model)
    X_df = pd.DataFrame(X, columns=df_model.columns)

    # Elbow curve (WCSS)
    wcss = []
    for k in K_ELBOW_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_df)
        wcss.append(km.inertia_)

    # Silhouette scores
    sil_scores = []
    for k in K_SIL_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_df)
        sil_scores.append(silhouette_score(X_df, labels))

    # Final model
    final_model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = final_model.fit_predict(X_df)
    final_sil = silhouette_score(X_df, labels)

    return scaler, final_model, labels, final_sil, list(K_ELBOW_RANGE), wcss, list(K_SIL_RANGE), sil_scores


@st.cache_data
def build_profiles(df: pd.DataFrame, labels):
    df = df.copy()
    df["cluster"] = labels
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["TotalPurchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["ChildrenAtHome"] = df["Kidhome"] + df["Teenhome"]

    profile = (
        df.groupby("cluster")
        .agg(
            Customers=("ID", "count"),
            AvgAge=("Age", "mean"),
            AvgIncome=("Income", "mean"),
            AvgTotalSpend=("TotalSpend", "mean"),
            AvgTotalPurchases=("TotalPurchases", "mean"),
            AvgRecency=("Recency", "mean"),
            AvgWebVisits=("NumWebVisitsMonth", "mean"),
            AvgDeals=("NumDealsPurchases", "mean"),
            AvgChildren=("ChildrenAtHome", "mean"),
            ResponseRate=("Response", "mean"),
            AvgWines=("MntWines", "mean"),
            AvgMeat=("MntMeatProducts", "mean"),
            AvgFruits=("MntFruits", "mean"),
            AvgFish=("MntFishProducts", "mean"),
            AvgSweets=("MntSweetProducts", "mean"),
            AvgGold=("MntGoldProds", "mean"),
            AvgWebPurchases=("NumWebPurchases", "mean"),
            AvgCatalogPurchases=("NumCatalogPurchases", "mean"),
            AvgStorePurchases=("NumStorePurchases", "mean"),
        )
        .reset_index()
    )

    sorted_spend = profile.sort_values("AvgTotalSpend")
    low_c = int(sorted_spend.iloc[0]["cluster"])
    high_c = int(sorted_spend.iloc[-1]["cluster"])
    mid_c = [int(c) for c in profile["cluster"] if c not in {low_c, high_c}][0]

    segment_map = {
        low_c: "Budget-Conscious Families",
        mid_c: "Digital Deal Seekers",
        high_c: "High-Value Loyalists",
    }
    profile["Segment"] = profile["cluster"].map(segment_map)
    df["Segment"] = df["cluster"].map(segment_map)

    return df, profile, segment_map


# ── Run pipeline ──────────────────────────────────────────────────────────────
df_raw, df_model, n_raw, n_missing = load_and_clean()
scaler, final_model, labels, final_sil, k_elbow, wcss, k_sil, sil_scores = build_pipeline(df_model)
df, profile, segment_map = build_profiles(df_raw, labels)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Customer Personality Segmentation")
st.caption(
    "K-Means clustering on 2,216 retail customers — showcasing the full ML pipeline "
    "from raw data through model selection to actionable segment insights."
)

m = st.columns(5)
m[0].metric("Customers", f"{len(df):,}")
m[1].metric("Features", len(df_model.columns))
m[2].metric("Clusters (k)", N_CLUSTERS)
m[3].metric("Avg. Income", f"${df['Income'].mean():,.0f}")
m[4].metric("Silhouette Score", f"{final_sil:.3f}")

st.divider()

tab_overview, tab_model, tab_profiles, tab_eda, tab_explorer, tab_predict = st.tabs([
    "Overview",
    "Model Building",
    "Cluster Profiles",
    "EDA",
    "Customer Explorer",
    "Segment Predictor",
])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        counts = df["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Customers"]
        fig = px.pie(
            counts, names="Segment", values="Customers",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="Segment Distribution",
            hole=0.35,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            df, x="Income", y="TotalSpend",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            hover_data=["Age", "TotalPurchases", "ChildrenAtHome"],
            title="Income vs. Total Spend by Segment",
            opacity=0.6,
        )
        fig.update_layout(xaxis_title="Annual Income ($)", yaxis_title="Total Spend ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Takeaways")
    insight_cols = st.columns(3)
    insights = [
        (
            "High-Value Loyalists", "#636EFA",
            "Highest income and spend across all categories. Strong campaign response rate. "
            "Prioritize retention, premium bundles, and early-access offers.",
        ),
        (
            "Digital Deal Seekers", "#EF553B",
            "Moderate spenders with high web activity and deal sensitivity. "
            "Use targeted web promotions and discount-led campaigns to grow basket size.",
        ),
        (
            "Budget-Conscious Families", "#00CC96",
            "Lower income, more children at home, lowest total spend. "
            "Value-focused messaging, essentials bundles, and loyalty programs will resonate most.",
        ),
    ]
    for col, (seg, color, text) in zip(insight_cols, insights):
        with col:
            st.markdown(
                f"<p style='color:{color}; font-weight:700; margin-bottom:4px'>{seg}</p>",
                unsafe_allow_html=True,
            )
            st.write(text)


# ── Tab 2: Model Building ─────────────────────────────────────────────────────
with tab_model:
    st.subheader("Full ML Pipeline")

    step_cols = st.columns(4)
    step_cols[0].info(
        "**1. Load & Clean**\n\n"
        f"{n_raw:,} raw customers loaded. {n_missing} rows with missing Income dropped → "
        f"**{len(df):,} customers** remain."
    )
    step_cols[1].info(
        "**2. Feature Engineering**\n\n"
        "`Age` computed from `Year_Birth`. 12 irrelevant or identifier columns dropped, leaving "
        f"**{len(df_model.columns)} features** for clustering."
    )
    step_cols[2].info(
        "**3. Scale**\n\n"
        "`StandardScaler` applied to all features. Each column transformed to mean=0, std=1 "
        "so no single feature dominates the distance metric."
    )
    step_cols[3].info(
        "**4. K-Means**\n\n"
        f"`k=3` chosen from elbow + silhouette analysis. `random_state=0`, `n_init=10`. "
        f"Final silhouette score: **{final_sil:.3f}**."
    )

    st.divider()

    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_elbow, y=wcss,
            mode="lines+markers",
            marker=dict(size=8, color="#636EFA"),
            line=dict(color="#636EFA", width=2),
            name="WCSS",
        ))
        fig.add_vline(x=3, line_dash="dash", line_color="red",
                      annotation_text="k=3 chosen", annotation_position="top right")
        fig.update_layout(
            title="Elbow Curve — WCSS vs Number of Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Within-Cluster Sum of Squares",
            xaxis=dict(tickmode="linear", dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_sil:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_sil, y=sil_scores,
            mode="lines+markers",
            marker=dict(size=8, color="#EF553B"),
            line=dict(color="#EF553B", width=2),
            name="Silhouette",
        ))
        fig.add_vline(x=3, line_dash="dash", line_color="red",
                      annotation_text="k=3 chosen", annotation_position="top right")
        fig.update_layout(
            title="Silhouette Score vs Number of Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            xaxis=dict(tickmode="linear", dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    sil_table = pd.DataFrame({"k": k_sil, "Silhouette Score": [round(s, 4) for s in sil_scores]})
    with st.expander("Silhouette score table"):
        st.dataframe(sil_table, hide_index=True, use_container_width=True)

    st.caption(
        "k=2 yields the highest silhouette score (0.28) but produces only two broad groups with limited "
        "business utility. k=3 gives a slightly lower score (0.21) while unlocking three meaningfully "
        "distinct segments — a worthwhile trade-off for marketing actionability."
    )

    st.divider()
    st.subheader(f"Feature Set — {len(df_model.columns)} columns used for clustering")
    feature_meta = {
        "Income": "Yearly household income ($)",
        "Kidhome": "Number of children at home",
        "Teenhome": "Number of teenagers at home",
        "Recency": "Days since last purchase",
        "MntWines": "Amount spent on wine ($)",
        "MntFruits": "Amount spent on fruits ($)",
        "MntMeatProducts": "Amount spent on meat ($)",
        "MntFishProducts": "Amount spent on fish ($)",
        "MntSweetProducts": "Amount spent on sweets ($)",
        "MntGoldProds": "Amount spent on gold products ($)",
        "NumDealsPurchases": "Purchases made using a discount",
        "NumWebPurchases": "Purchases through the website",
        "NumCatalogPurchases": "Purchases through catalog",
        "NumStorePurchases": "Purchases in-store",
        "NumWebVisitsMonth": "Website visits in the last month",
        "Complain": "Complained in the last 2 years (0/1)",
        "Response": "Accepted the last campaign (0/1)",
        "Age": "Age derived from Year_Birth",
    }
    feat_df = pd.DataFrame(
        [{"Feature": k, "Description": v} for k, v in feature_meta.items()
         if k in df_model.columns]
    )
    st.dataframe(feat_df, hide_index=True, use_container_width=True)


# ── Tab 3: Cluster Profiles ───────────────────────────────────────────────────
with tab_profiles:
    st.subheader("Segment Summary")

    summary_cols = {
        "Segment": "Segment",
        "Customers": "Customers",
        "AvgAge": "Avg Age",
        "AvgIncome": "Avg Income",
        "AvgTotalSpend": "Avg Total Spend",
        "AvgTotalPurchases": "Avg Purchases",
        "AvgWebVisits": "Avg Web Visits/mo",
        "AvgDeals": "Avg Deal Purchases",
        "AvgChildren": "Avg Children",
        "ResponseRate": "Campaign Response",
    }
    disp = (
        profile[list(summary_cols.keys())]
        .rename(columns=summary_cols)
        .sort_values("Avg Total Spend", ascending=False)
    )
    st.dataframe(
        disp.style.format({
            "Avg Income": "${:,.0f}",
            "Avg Total Spend": "${:,.0f}",
            "Avg Age": "{:.0f}",
            "Avg Purchases": "{:.1f}",
            "Avg Web Visits/mo": "{:.1f}",
            "Avg Deal Purchases": "{:.1f}",
            "Avg Children": "{:.1f}",
            "Campaign Response": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    col_radar, col_channel = st.columns(2)

    with col_radar:
        st.subheader("Spending by Category")
        categories = ["Wines", "Meat", "Fish", "Fruits", "Sweets", "Gold"]
        cat_map = {
            "Wines": "AvgWines", "Meat": "AvgMeat", "Fish": "AvgFish",
            "Fruits": "AvgFruits", "Sweets": "AvgSweets", "Gold": "AvgGold",
        }
        fig = go.Figure()
        for _, row in profile.iterrows():
            seg = row["Segment"]
            vals = [row[cat_map[c]] for c in categories]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=seg,
                line_color=SEGMENT_COLORS[seg],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
            title="Average $ Spent per Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_channel:
        st.subheader("Purchase Channels")
        channels = {
            "Web": "AvgWebPurchases",
            "Catalog": "AvgCatalogPurchases",
            "Store": "AvgStorePurchases",
            "Deals": "AvgDeals",
        }
        ch_rows = []
        for _, row in profile.iterrows():
            for ch, col in channels.items():
                ch_rows.append({"Segment": row["Segment"], "Channel": ch, "Avg Purchases": row[col]})
        ch_df = pd.DataFrame(ch_rows)
        fig = px.bar(
            ch_df, x="Channel", y="Avg Purchases", color="Segment",
            barmode="group", color_discrete_map=SEGMENT_COLORS,
            title="Avg Purchases by Channel",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Distribution by Variable")
    compare_var = st.selectbox("Select variable", [
        "Income", "TotalSpend", "TotalPurchases", "Age", "Recency",
        "NumWebVisitsMonth", "MntWines", "MntMeatProducts", "ChildrenAtHome",
    ])
    fig = px.box(
        df, x="Segment", y=compare_var,
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        points="outliers",
        title=f"{compare_var} Distribution by Segment",
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: EDA ────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Correlation Analysis")
    st.caption("Pairs with ≥ 0.6 positive correlation were highlighted in the original notebook analysis.")

    numeric_cols = df_model.select_dtypes(include=np.number).columns.tolist()
    corr = df_model[numeric_cols].corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Key Correlations Found (≥ 0.60)")
    strong_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if v >= 0.6:
                strong_pairs.append({"Feature A": cols[i], "Feature B": cols[j], "Correlation": round(v, 3)})
    if strong_pairs:
        st.dataframe(
            pd.DataFrame(strong_pairs).sort_values("Correlation", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Feature Distributions")
    dist_var = st.selectbox("Select feature", numeric_cols, key="eda_dist")
    fig = px.histogram(
        df_model, x=dist_var, nbins=40,
        title=f"Distribution of {dist_var}",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Customer Explorer ──────────────────────────────────────────────────
with tab_explorer:
    f1, f2 = st.columns(2)
    with f1:
        selected_segs = st.multiselect(
            "Filter by segment",
            options=sorted(df["Segment"].unique()),
            default=sorted(df["Segment"].unique()),
        )
    with f2:
        income_lo = int(df["Income"].min())
        income_hi = int(df["Income"].quantile(0.99))
        income_range = st.slider(
            "Income range ($)",
            min_value=income_lo,
            max_value=income_hi,
            value=(int(df["Income"].quantile(0.05)), income_hi),
            step=1000,
        )

    filtered = df[
        df["Segment"].isin(selected_segs)
        & df["Income"].between(income_range[0], income_range[1])
    ]

    st.metric("Matching customers", f"{len(filtered):,}")
    st.dataframe(
        filtered[[
            "ID", "Segment", "Age", "Income", "ChildrenAtHome",
            "TotalSpend", "TotalPurchases", "NumWebVisitsMonth", "Recency", "Response",
        ]].sort_values("TotalSpend", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 6: Segment Predictor ──────────────────────────────────────────────────
with tab_predict:
    st.subheader("Predict a New Customer's Segment")
    st.caption(
        "Uses the same StandardScaler and K-Means model fit on the training data. "
        "Adjust the inputs below and the prediction updates instantly."
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**Demographics & Behavior**")
        p_income = st.number_input("Income ($)", 0, 700_000, 52_000, 1_000)
        p_age = st.number_input("Age", 18, 100, 45)
        p_kidhome = st.number_input("Kids at home", 0, 3, 0)
        p_teenhome = st.number_input("Teens at home", 0, 3, 0)
        p_recency = st.number_input("Days since last purchase", 0, 99, 40)
        p_complain = st.toggle("Complained in last 2 years")
        p_response = st.toggle("Accepted last campaign")

    with pc2:
        st.markdown("**Spending ($)**")
        p_wines = st.number_input("Wines", 0, 1_500, 200)
        p_fruits = st.number_input("Fruits", 0, 200, 10)
        p_meat = st.number_input("Meat", 0, 1_800, 100)
        p_fish = st.number_input("Fish", 0, 260, 20)
        p_sweets = st.number_input("Sweets", 0, 265, 10)
        p_gold = st.number_input("Gold", 0, 365, 25)

    with pc3:
        st.markdown("**Purchases by Channel**")
        p_deals = st.number_input("Deal purchases", 0, 15, 2)
        p_web = st.number_input("Web purchases", 0, 27, 4)
        p_catalog = st.number_input("Catalog purchases", 0, 28, 2)
        p_store = st.number_input("Store purchases", 0, 13, 5)
        p_webvisits = st.number_input("Monthly web visits", 0, 20, 5)

    input_dict = {
        "Income": p_income, "Kidhome": p_kidhome, "Teenhome": p_teenhome,
        "Recency": p_recency, "MntWines": p_wines, "MntFruits": p_fruits,
        "MntMeatProducts": p_meat, "MntFishProducts": p_fish,
        "MntSweetProducts": p_sweets, "MntGoldProds": p_gold,
        "NumDealsPurchases": p_deals, "NumWebPurchases": p_web,
        "NumCatalogPurchases": p_catalog, "NumStorePurchases": p_store,
        "NumWebVisitsMonth": p_webvisits, "Complain": int(p_complain),
        "Response": int(p_response), "Age": p_age,
    }
    input_row = pd.DataFrame([input_dict])[df_model.columns]

    scaled_input = scaler.transform(input_row)
    raw_cluster = int(final_model.predict(scaled_input)[0])
    predicted_segment = segment_map[raw_cluster]
    color = SEGMENT_COLORS[predicted_segment]

    st.markdown("---")
    st.markdown(
        f"<div style='background:{color}18; border-left:5px solid {color}; "
        f"padding:1rem 1.25rem; border-radius:6px'>"
        f"<span style='font-size:0.85rem; color:#888'>Predicted Segment</span><br>"
        f"<span style='font-size:1.5rem; font-weight:700; color:{color}'>{predicted_segment}</span>"
        f"<span style='font-size:0.8rem; color:#aaa; margin-left:1rem'>Raw cluster label: {raw_cluster}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**How this customer compares to the segment average:**")
    seg_row = profile[profile["cluster"] == raw_cluster].iloc[0]
    p_total_spend = p_wines + p_fruits + p_meat + p_fish + p_sweets + p_gold
    p_total_purchases = p_deals + p_web + p_catalog + p_store
    comparison = pd.DataFrame([
        {"Metric": "Income ($)", "This Customer": p_income, "Segment Avg": seg_row["AvgIncome"]},
        {"Metric": "Total Spend ($)", "This Customer": p_total_spend, "Segment Avg": seg_row["AvgTotalSpend"]},
        {"Metric": "Total Purchases", "This Customer": p_total_purchases, "Segment Avg": seg_row["AvgTotalPurchases"]},
        {"Metric": "Web Visits/mo", "This Customer": p_webvisits, "Segment Avg": seg_row["AvgWebVisits"]},
        {"Metric": "Deal Purchases", "This Customer": p_deals, "Segment Avg": seg_row["AvgDeals"]},
        {"Metric": "Children at Home", "This Customer": p_kidhome + p_teenhome, "Segment Avg": seg_row["AvgChildren"]},
    ])
    st.dataframe(
        comparison.style.format({"This Customer": "{:,.0f}", "Segment Avg": "{:,.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
