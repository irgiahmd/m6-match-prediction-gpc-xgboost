# m6-match-prediction-gpc-xgboost
Comparative analysis of Gaussian Process Classifier (GPC) and XGBoost in predicting match outcomes using Mobile Legends M6 dataset.
# 🧠 M6 Match Prediction using GPC & XGBoost

This repository contains the implementation of a comparative analysis between Gaussian Process Classifier (GPC) and XGBoost in predicting match outcomes in the **Mobile Legends M6 World Championship** dataset.

## 📌 Project Overview
The objective of this research is to compare the performance of two classification algorithms — GPC and XGBoost — on esports data, specifically Mobile Legends M6 match results. Evaluation is based on metrics such as **Accuracy**, **F1-Score**, **Confusion Matrix**, and **ROC Curve**.


## 📊 Algorithms Used
- Gaussian Process Classifier (GPC)
- Extreme Gradient Boosting (XGBoost)

## ⚙️ How to Run

1. git clone https://github.com/irgiahmad/m6-match-prediction-gpc-xgboost.git

2. Install required libraries:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py


## 📈 Evaluation Metrics
- Accuracy
- F1 Score
- Confusion Matrix
- ROC Curve

## 📌 Results
- XGBoost outperformed GPC in both accuracy and F1-score.
- GPC showed more conservative predictions with higher bias.
- Dataset quality and class imbalance also influenced the results.

📊 Mobile Legends M6 Match Statistics
In addition to model analysis, this project also presents comprehensive statistics from the Mobile Legends M6 Championship, including:

🔸 Team Performance
Win rate per team

Most picked and banned heroes

Match count and stage distribution (Swiss vs Knockout)

🔸 Hero Statistics
Top 5 most picked heroes

Top 5 most banned heroes

Hero win rate distribution

🔸 Player Performance
Top MVPs per stage

Best players based on KDA and MVP score

Player performance by role (e.g., EXP, Jungle, Mid, Gold, Roamer)

📈 Visualizations
📉 Bar chart of team win rates

🛡️ Hero ban/pick trends (bar and pie charts)

🧠 MVP analysis per stage (table)

🧮 Player ranking by role (interactive tables)

🌀 Heatmaps and violin plots for deeper distribution analysis

These visualizations aim to give insight into in-game meta, team strategies, and standout player performances during the M6 World Championship.

## 👨‍💻 Author
**Irgi Ahmad Alfarizi**  
Fresh graduate in Computer Science, Universitas Sumatera Utara  
Email: [irgiahmadalfarizi14@gmail.com)


