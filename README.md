# 📱 Customer Churn Prediction

> Predicting which telecom customers will leave — and why — using XGBoost, SHAP, and Streamlit.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://customer-churn-prediction-scoxvfmcnhs84xzcqxjqnh.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-vin1bun-181717?style=for-the-badge&logo=github)](https://github.com/vin1bun/customer-churn-prediction)

---

## 🎯 Business Problem

Retaining a customer costs **5x less** than acquiring a new one.

For a telecom company, even a 1% reduction in churn can save crores annually. The goal of this project was to identify **which customers are likely to leave before they actually do** — giving the retention team enough time to intervene with targeted offers.

---

## 📊 Dataset

| Item | Detail |
|------|--------|
| Source | IBM Telco Customer Churn (Kaggle) |
| Rows | 7,043 customers |
| Features | 20 (after cleaning) |
| Target | Churn → Yes / No |
| Class Balance | 73% Stayed / 27% Churned |

**Key Data Challenge:**
`TotalCharges` was stored as text (object) instead of numeric — caused by 11 rows with blank spaces for brand new customers with tenure = 0. Detected via data type inspection, fixed using `pd.to_numeric(errors='coerce')`, and rows dropped as they represented only 0.15% of data.

---

## 🔍 EDA Insights

Before touching any model, EDA revealed clear churn signals:

| Feature | Insight |
|---------|---------|
| Contract Type | Month-to-month customers churned at **42%** vs 3% for two-year contracts |
| Tenure | Customers in first **12 months** had ~50% churn rate |
| Payment Method | Electronic check users churned at **45%** vs 15% for auto payments |
| Internet Service | Fiber optic customers had **42% churn** despite being premium users |
| Senior Citizens | Churned at **2x the rate** of younger customers |
| Gender | Almost **zero impact** on churn — both genders identical |

---

## ⚙️ Feature Engineering

Created 4 business-driven features — 2 of which ranked in SHAP Top 10:

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| `avg_monthly_spend` | TotalCharges / tenure | Normalized customer value |
| `services_count` | Count of add-on services | Engagement = loyalty |
| `is_new_customer` | tenure < 12 → 1 | Flag highest risk segment |
| `charge_per_service` | MonthlyCharges / services_count | Value for money signal |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Imbalanced-learn, Optuna
- **Deployment:** Streamlit Cloud
- **Environment:** Google Colab

---

## 🤖 Model Building

Trained 3 models and compared performance:

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.815 |
| Random Forest | 0.807 |
| XGBoost (default) | 0.819 |
| **XGBoost (Optuna tuned)** | **0.829** ✅ |

**Key Decisions:**
- Used **SMOTE** on training data only to handle 73-27 class imbalance
- Used **StandardScaler** fit on training data only (no data leakage)
- Used **Optuna** (50 trials) for hyperparameter tuning — best params: n_estimators=174, max_depth=3, learning_rate=0.022
- Used **drop_first=True** in One Hot Encoding to avoid dummy variable trap

---

## 📈 Evaluation

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.829 |
| Recall (Churners) | 69.5% |
| Precision (Churners) | 52% |
| F1 Score (Churners) | 0.60 |

**Confusion Matrix:**
```
                 Predicted
                 Stayed   Churned
Actual  Stayed    797       236
        Churned   114       260 ✅
```

> Prioritized **Recall over Precision** — missing a churner (False Negative) costs far more than a false alarm. Each missed churner = lost revenue with no retention opportunity.

## 💰 Business Impact — Customer Lifetime Value (CLV)

Beyond predicting churn, I calculated the business value at risk for each customer:

| Metric | Value |
|--------|-------|
| Total CLV at Risk | $714,578 |
| Average CLV per Churner | $1,247 |
| Monthly Revenue at Risk | $41,625 |
| High Risk Customers | 573 customers |

**Why CLV matters:**
Not all churners are equal. A customer paying $95/month deserves more
retention effort than one paying $25/month. CLV helps the retention team
decide WHERE to spend their budget — prioritizing high value customers first.

**Priority Tiers in the Streamlit App:**
- 🔴 HIGH PRIORITY → CLV > $1,500
- 🟡 MEDIUM PRIORITY → CLV $800–$1,500
- 🟢 LOW PRIORITY → CLV < $800
---

## 🔎 SHAP Explainability

SHAP validated every EDA hypothesis:

| Rank | Feature | Direction |
|------|---------|-----------|
| 1 | tenure | Low tenure → High churn risk |
| 2 | Contract_Two year | Has 2yr contract → Stays |
| 3 | PaymentMethod_Electronic check | E-check → Churns |
| 4 | InternetService_Fiber optic | Fiber → Churns more |
| 5 | charge_per_service ⭐ | High charge/service → Churns |

⭐ Engineered feature that outranked original columns

---

## 🚀 Streamlit App

**Live App:** [customer-churn-prediction.streamlit.app](https://customer-churn-prediction-scoxvfmcnhs84xzcqxjqnh.streamlit.app/)

The app doesn't just predict churn — it gives **actionable retention recommendations:**

- Month-to-month customer → Offer discounted annual contract
- Electronic check user → Encourage automatic payment switch
- New customer (< 12 months) → Trigger onboarding support program
- Few services → Offer bundled services discount

---

## 📂 Project Structure

```
customer-churn-prediction/
├── app.py                  # Streamlit application
├── model.pkl               # Trained XGBoost model
├── scaler.pkl              # Fitted StandardScaler
├── feature_names.pkl       # Feature order for prediction
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🏃 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/vin1bun/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🔮 Future Improvements

- Add **SHAP waterfall plot** for individual customer explanation in the app
- Compare **SMOTE vs class_weight** approach systematically
- Add **cost matrix** to quantify exact revenue saved by retention
- Experiment with **LightGBM** as an alternative to XGBoost

---

## 👤 Author

**Vineet** — Aspiring Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vineetprakash03-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/vineetprakash03)
[![GitHub](https://img.shields.io/badge/GitHub-vin1bun-181717?style=flat&logo=github)](https://github.com/vin1bun)
[![Kaggle](https://img.shields.io/badge/Kaggle-vin1bun-20BEFF?style=flat&logo=kaggle)](https://kaggle.com/vin1bun)

---

> *"Models find correlation, not causation — but actionable retention strategy starts with knowing who is at risk."*
