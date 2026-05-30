# Customer Personality Segmentation

An end-to-end machine learning project that segments 2,216 retail customers into three actionable groups using K-Means clustering — wrapped in an interactive Streamlit dashboard built for portfolio presentation.

---

## What This Project Does

Retail companies collect rich behavioral data on their customers but often struggle to turn it into targeted strategy. This project solves that by:

1. **Cleaning and preparing** raw customer data (demographics, spending, purchase channels, campaign history)
2. **Engineering features** such as customer age and total spend across six product categories
3. **Selecting the optimal cluster count** using the elbow method and silhouette scores across k=2 through k=10
4. **Fitting a K-Means model** (k=3, StandardScaler normalization) to identify three distinct customer segments
5. **Profiling each segment** to extract actionable business insights
6. **Deploying the full pipeline** as an interactive web dashboard

---

## The Three Segments

| Segment | Description |
|---|---|
| **High-Value Loyalists** | Highest income and total spend, strong campaign response. Prioritize retention and premium offers. |
| **Digital Deal Seekers** | Moderate spenders, high web activity, deal-sensitive. Convert with targeted online promotions. |
| **Budget-Conscious Families** | Lower income, more children at home, value-focused. Engage with essentials bundles and loyalty perks. |

---

## Dashboard Features

The Streamlit app is organized into six tabs:

- **Overview** — Segment distribution, income vs. spend scatter, business takeaways
- **Model Building** — Elbow curve, silhouette score plot, full pipeline walkthrough, feature table
- **Cluster Profiles** — Spending radar chart, purchase channel breakdown, per-variable box plots
- **EDA** — Correlation heatmap, high-correlation pairs (≥0.60), feature distributions
- **Customer Explorer** — Filter customers by segment and income range, browse individual records
- **Segment Predictor** — Enter any customer's attributes and get a live segment prediction with a comparison to the segment average

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading, cleaning, feature engineering |
| `scikit-learn` | StandardScaler, KMeans, silhouette_score |
| `plotly` | Interactive charts (scatter, box, radar, heatmap) |
| `streamlit` | Dashboard framework and deployment |

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/TahaAsim-ui/Customer-Personality-Segmentation.git
cd Customer-Personality-Segmentation

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Data

`Customer_Personality_Segmentation.csv` — 2,240 customers with 29 attributes covering demographics, six spending categories, four purchase channels, five marketing campaign responses, and household composition. Source: [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

**Preprocessing applied:**
- 24 rows with missing `Income` dropped → 2,216 customers retained
- 12 identifier/irrelevant columns removed
- `Age` derived from `Year_Birth`
- All 18 remaining features scaled with `StandardScaler` before clustering

---

## Key Findings

- **MntMeatProducts × NumCatalogPurchases** — strongest positive correlation (0.73)
- **MntWines × NumStorePurchases** — second strongest (0.64)
- Silhouette score at k=3: **0.213** — k=2 scores higher (0.28) but yields two overly broad groups with less marketing utility
- High-Value Loyalists make up ~26% of customers but drive a disproportionate share of revenue

---

## Deploy for Free

Host this dashboard publicly on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Fork this repository
2. Go to Streamlit Community Cloud → New app
3. Select this repo, branch `main`, file `app.py`
4. Deploy — add the live URL to your resume and LinkedIn
