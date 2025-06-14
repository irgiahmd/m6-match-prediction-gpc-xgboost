import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost as xgb
import plotly.figure_factory as ff
import time

# Konfigurasi halaman
st.set_page_config(page_title="Analisis GPC vs XGBoost - M6", layout="wide")
st.title("📊 Analisis Perbandingan GPC vs XGBoost - Mobile Legends World Championship 6")

# Upload dataset
st.sidebar.header("📂 Upload Dataset CSV")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')
    st.success("✅ Dataset berhasil dimuat!")

    # Normalisasi nama kolom
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Buat kolom is_win
    if 'team_name' in df.columns and 'winner' in df.columns:
        df["is_win"] = (df["team_name"] == df["winner"]).astype(int)
    else:
        st.error("❌ Kolom 'team_name' atau 'winner' tidak ditemukan dalam dataset.")
        st.stop()

    # Bersihkan nilai kosong penting
    df.dropna(subset=["hero", "role", "kill", "death", "assist"], inplace=True)

    # Label Encoding
    df["hero_encoded"] = LabelEncoder().fit_transform(df["hero"].astype(str))
    df["role_encoded"] = LabelEncoder().fit_transform(df["role"].astype(str))

    # Siapkan fitur dan target
    X = df[["hero_encoded", "kill", "death", "assist", "role_encoded"]]
    y = df["is_win"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Pelatihan model
    model_gpc = GaussianProcessClassifier().fit(X_train, y_train)
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)

    # Prediksi
    y_pred_gpc = model_gpc.predict(X_test)
    y_pred_xgb = model_xgb.predict(X_test)

    # Evaluasi GPC
    if len(np.unique(y_pred_gpc)) < 2:
        acc_gpc = 0.0
        f1_gpc = 0.0
        st.warning("⚠️ Model GPC hanya memprediksi satu kelas (0/1), metrik tidak dapat dihitung.")
    else:
        acc_gpc = accuracy_score(y_test, y_pred_gpc)
        f1_gpc = f1_score(y_test, y_pred_gpc)

    # Evaluasi XGBoost
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)

    # Tampilan metrik utama
    st.header("🎯 Evaluasi Model & Penjelasan")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi GPC", f"{acc_gpc:.2f}")
        st.metric("F1 Score GPC", f"{f1_gpc:.2f}")
    with col2:
        st.metric("Akurasi XGBoost", f"{acc_xgb:.2f}")
        st.metric("F1 Score XGBoost", f"{f1_xgb:.2f}")

    # Waktu training
    start_gpc = time.time()
    model_gpc.fit(X_train, y_train)
    gpc_time = time.time() - start_gpc

    start_xgb = time.time()
    model_xgb.fit(X_train, y_train)
    xgb_time = time.time() - start_xgb

    st.metric("⏱️ Waktu Training GPC", f"{gpc_time:.2f} detik")
    st.metric("⏱️ Waktu Training XGBoost", f"{xgb_time:.2f} detik")

    # Tabel Perbandingan
    st.subheader("📋 Tabel Perbandingan GPC dan XGBoost")
    table_data = pd.DataFrame({
        "Model": ["GPC", "XGBoost"],
        "Akurasi": [acc_gpc, acc_xgb],
        "F1 Score": [f1_gpc, f1_xgb]
    })
    table_data.index = range(1, len(table_data) + 1)
    st.dataframe(table_data, use_container_width=True)

    # Grafik Perbandingan
    st.subheader("📊 Grafik Perbandingan F1 Score dan Akurasi")
    fig_bar = px.bar(table_data.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                     x="Model", y="Score", color="Metric", barmode="group", text_auto=".2f")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    cm_gpc = confusion_matrix(y_test, y_pred_gpc)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)

    st.markdown("**GPC Confusion Matrix**")
    st.dataframe(pd.DataFrame(cm_gpc, columns=["Pred Negative", "Pred Positive"], index=["Actual Negative", "Actual Positive"]))

    st.markdown("**XGBoost Confusion Matrix**")
    st.dataframe(pd.DataFrame(cm_xgb, columns=["Pred Negative", "Pred Positive"], index=["Actual Negative", "Actual Positive"]))
    
    # ROC Curve & AUC
    st.subheader("📈 ROC Curve & AUC Score")
    y_prob_gpc = model_gpc.predict_proba(X_test)[:, 1]
    y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

    fpr_gpc, tpr_gpc, _ = roc_curve(y_test, y_prob_gpc)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_gpc = auc(fpr_gpc, tpr_gpc)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr_gpc, tpr_gpc, label=f"GPC (AUC = {auc_gpc:.2f})", color="blue")
    ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.2f})", color="green")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - GPC vs XGBoost")
    ax.legend()
    st.pyplot(fig_roc)

    # Penjelasan AUC
    st.markdown(f"""
    - **AUC GPC**: {auc_gpc:.2f}  
    - **AUC XGBoost**: {auc_xgb:.2f}
    """)

    # Hasil Encoding
    st.subheader("🔢 Contoh Hasil Encoding Fitur Kategorikal")
    preview = df[["hero", "hero_encoded", "role", "role_encoded"]].drop_duplicates().head(10)
    preview.index += 1
    st.dataframe(preview, use_container_width=True)


   # ------------------ PEMISAH UTAMA ------------------
    st.markdown("---")
    st.header("🎮 Statistik Mobile Legends World Championship 6")
    st.markdown("Analisis lengkap mulai dari tim pemenang, hero pick/ban, hingga performa individu pemain berdasarkan KDA dan MVP.")
    # Statistik Tim
    st.subheader("📊 Statistik Tim: Total Match, Total Win, dan Winrate")
    match_df = df.drop_duplicates(subset=["match_id", "team_name"])
    team_stats = match_df.groupby("team_name").agg(
        Total_Match=("match_id", "count"), Total_Win=("is_win", "sum"))
    team_stats["Winrate (%)"] = (team_stats["Total_Win"] / team_stats["Total_Match"]) * 100
    # Urutkan berdasarkan Winrate tertinggi
    team_stats = team_stats.sort_values("Winrate (%)", ascending=False).head(16).reset_index()
    team_stats.index = range(1, len(team_stats) + 1)
    st.dataframe(team_stats, use_container_width=True)
    st.subheader("🏅 Top 5 Pemain MVP Terbanyak")
    if "mvp_status" in df.columns and "player_name" in df.columns:
        df["mvp_status"] = df["mvp_status"].astype(str).str.strip().str.lower()
        mvp_df = df[df["mvp_status"] == "mvp"]
        if not mvp_df.empty:
            top_mvp = mvp_df.groupby(["player_name", "team_name", "role"]).size().reset_index(name="MVP Count")
            top_mvp = top_mvp.sort_values("MVP Count", ascending=False).head(5).reset_index(drop=True)
            top_mvp.index = range(1, len(top_mvp) + 1)
            top_mvp = top_mvp.rename(columns={
                "player_name": "Player", "team_name": "Team", "role": "Role"
            })
            st.dataframe(top_mvp[["Player", "Team", "Role", "MVP Count"]], use_container_width=True)
        else:
            st.info("Tidak ada pemain dengan status MVP yang ditemukan.")
    else:
        st.warning("Kolom 'mvp_status' atau 'player_name' tidak tersedia.")

     # Statistik Hero Pick
    st.subheader("🔥 Top 10 Hero yang Paling Sering di-Pick")
    pick_count = df['hero'].value_counts().head(10).reset_index()
    pick_count.columns = ['Hero', 'Pick Count']
    pick_count.index = range(1, len(pick_count) + 1)
    st.dataframe(pick_count, use_container_width=True)

    # Grafik Hero Pick
    fig_pick = px.bar(pick_count, x='Hero', y='Pick Count', title="Grafik Hero yang Paling Sering di-Pick",
                      color='Pick Count', text_auto=True)
    st.plotly_chart(fig_pick, use_container_width=True)

    # Winrate Hero
    st.subheader("📊 Statistik Hero: Pick, Menang, dan Winrate")
    hero_stats = df.groupby("hero").agg(Pick_Count=("hero", "count"), Win_Count=("is_win", "sum"))
    hero_stats["Winrate (%)"] = (hero_stats["Win_Count"] / hero_stats["Pick_Count"]) * 100
    hero_stats = hero_stats.sort_values("Pick_Count", ascending=False).head(10).reset_index()
    hero_stats.index = range(1, len(hero_stats) + 1)
    st.dataframe(hero_stats, use_container_width=True)

    # Statistik Hero Ban berdasarkan match_id
    st.subheader("🚫 Top 10 Hero yang Paling Sering di-Ban")
    df['hero_ban'] = df['hero_ban'].fillna('')
    ban_df = df[['match_id', 'hero_ban']].drop_duplicates()
    ban_expanded = ban_df['hero_ban'].str.split(',').explode().str.strip()
    ban_count = ban_expanded[ban_expanded != ''].value_counts().head(10)
    ban_df_top = ban_count.reset_index()
    ban_df_top.columns = ['Hero', 'Ban Count']
    ban_df_top.index = range(1, len(ban_df_top) + 1)
    st.dataframe(ban_df_top, use_container_width=True)

    # Grafik Hero Ban
    fig_ban = px.bar(ban_df_top, x='Hero', y='Ban Count', title="Grafik Hero yang Paling Sering di-Ban",
                     color='Ban Count', text_auto=True)
    st.plotly_chart(fig_ban, use_container_width=True)

    # Pemain Terbaik per Role (Top 5 per role berdasarkan KDA)
    st.subheader("🏅 Top 5 Pemain Terbaik per Role (Berdasarkan KDA)")
    df["KDA"] = (df["kill"] + df["assist"]) / df["death"].replace(0, 1)
    top_kda_role = df.groupby(["role", "player_name", "team_name"])["KDA"].mean().reset_index()
    top_kda_role = top_kda_role.sort_values(["role", "KDA"], ascending=[True, False])

    for role in top_kda_role["role"].unique():
        st.markdown(f"#### ⚔️ Role: {role}")
        top5 = top_kda_role[top_kda_role["role"] == role].head(5).reset_index(drop=True)
        top5.index += 1
        top5.columns = ["Role", "Player", "Team", "Avg KDA"]
        st.dataframe(top5[["Player", "Team", "Avg KDA"]], use_container_width=True)
    # Pemain terbaik per stage
    st.subheader("👤 Pemain Terbaik per Stage (Swiss & Knockout)")
    if "stage" in df.columns:
        for stage in df["stage"].dropna().unique():
            st.markdown(f"### 🏆 Stage: {stage}")
            stage_df = df[df["stage"] == stage].copy()
            stage_df["KDA"] = (stage_df["kill"] + stage_df["assist"]) / stage_df["death"].replace(0, 1)
            stage_df["mvp_numeric"] = stage_df["mvp_status"].str.contains("mvp", case=False, na=False).astype(int)

            player_scores = stage_df.groupby(["player_name", "team_name"]).agg(
                MVP_Total=("mvp_numeric", "sum"),
                KDA_Mean=("KDA", "mean")
            ).reset_index()
            player_scores["Score"] = player_scores["MVP_Total"] * 0.7 + player_scores["KDA_Mean"] * 0.3
            best_player = player_scores.sort_values("Score", ascending=False).iloc[0]

            top_ban = stage_df['hero_ban'].str.split(',').explode().str.strip().value_counts().idxmax()
            top_pick = stage_df['hero'].value_counts().idxmax()

            st.markdown(f"- 👑 **Pemain Terbaik**: `{best_player['player_name']}` dari {best_player['team_name']} (Score: {best_player['Score']:.2f})")
            st.markdown(f"- ❌ **Hero Paling Sering di-Ban**: `{top_ban}`")
            st.markdown(f"- ✅ **Hero Paling Sering di-Pick**: `{top_pick}`")
    else:
        st.warning("Kolom 'stage' tidak tersedia.")
        
    # Kesimpulan
    st.subheader("📝 Kesimpulan")
    st.success(f"""
✅ **Kesimpulan Akhir:**

- 🔍 **Model terbaik berdasarkan F1 Score**: **{'XGBoost' if f1_xgb > f1_gpc else 'GPC'}**
- 📊 **F1 Score** lebih unggul sebagai metrik evaluasi karena mempertimbangkan keseimbangan antara **precision** dan **recall**, sehingga memberikan gambaran yang adil, terutama saat data tidak seimbang.
- 📉 **ROC Curve** memperlihatkan bahwa XGBoost memiliki **AUC (Area Under Curve)** yang lebih tinggi, artinya model ini lebih baik dalam membedakan antara tim yang menang dan kalah.
- ⚔️ Statistik **hero pick** dan **hero ban** mengungkapkan pola meta dan strategi yang digunakan oleh tim-tim di turnamen.
- 🧙 **Pemain terbaik** dihitung berdasarkan gabungan **MVP** dan **KDA** dengan bobot 70% MVP dan 30% KDA, untuk menilai kontribusi individual pemain secara lebih adil.
- 🗺️ Analisis berdasarkan **stage turnamen (Swiss & Knockout)** memberikan pemahaman mendalam terhadap perbedaan strategi dan performa tim dalam berbagai tekanan kompetisi.
""")


    st.caption("Skripsi oleh Irgi Ahmad Alfarizi - Universitas Sumatera Utara")
