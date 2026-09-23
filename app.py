import streamlit as st
import pandas as pd
import joblib


# Load model 

@st.cache_resource
def load_artifacts():
    model = joblib.load('final_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    numeric_cols = joblib.load('numeric_cols.pkl')
    category_cols = joblib.load('category_cols.pkl')
    return model, scaler, encoder, feature_columns, numeric_cols, category_cols

model, scaler, encoder, feature_columns, numeric_cols, category_cols = load_artifacts()

# page
st.set_page_config(page_title="F1 Race Result Predictor", layout="centered")

st.title("F1 Race Result Predictor")
st.markdown(
    "Aplikasi ini memprediksi kategori hasil finish seorang pembalap "
    "(**Podium**, **Poin**, atau **Non Poin**) berdasarkan data sebelum balapan dimulai."
)
st.divider()

# ambil daftar kategori unik dari encoder 
kategori_per_kolom = dict(zip(category_cols, encoder.categories_))


# form input dari puser
st.subheader("Masukkan Data Race")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Tahun", min_value=1950, max_value=2100, value=2024, step=1)
    round_ = st.number_input("Round (urutan race di musim)", min_value=1, max_value=30, value=1, step=1)
    grid = st.number_input("Posisi Start (Grid)", min_value=0, max_value=30, value=1, step=1)
    circuit_altitude = st.number_input("Ketinggian Sirkuit (meter)", min_value=0, max_value=3000, value=100, step=1)

with col2:
    circuit_name = st.selectbox("Nama Sirkuit", sorted(kategori_per_kolom['circuit_name']))
    circuit_country = st.selectbox("Negara Sirkuit", sorted(kategori_per_kolom['circuit_country']))
    driver_name = st.selectbox("Nama Pembalap", sorted(kategori_per_kolom['driver_name']))
    driver_nationality = st.selectbox("Kebangsaan Pembalap", sorted(kategori_per_kolom['driver_nationality']))

col3, col4 = st.columns(2)
with col3:
    constructor_name = st.selectbox("Nama Tim/Konstruktor", sorted(kategori_per_kolom['constructor_name']))
with col4:
    constructor_nationality = st.selectbox("Kebangsaan Tim", sorted(kategori_per_kolom['constructor_nationality']))

st.markdown("**Data performa (opsional, isi jika ada estimasi):**")
col5, col6 = st.columns(2)
with col5:
    avg_lap_time_ms = st.number_input("Rata-rata Waktu Lap (milidetik)", min_value=0, value=90000, step=100)
with col6:
    fastest_lap_time_ms = st.number_input("Waktu Lap Tercepat (milidetik)", min_value=0, value=88000, step=100)

st.divider()

# predict button
if st.button("🔮 Prediksi Hasil Finish", type="primary"):

    data_baru = pd.DataFrame([{
        'year': year,
        'round': round_,
        'circuit_name': circuit_name,
        'circuit_country': circuit_country,
        'circuit_altitude': circuit_altitude,
        'driver_name': driver_name,
        'driver_nationality': driver_nationality,
        'constructor_name': constructor_name,
        'constructor_nationality': constructor_nationality,
        'grid': grid,
        'avg_lap_time_ms': avg_lap_time_ms,
        'fastest_lap_time_ms': fastest_lap_time_ms
    }])

    # numerik & kategorikal kolom
    data_num = data_baru[numeric_cols]
    data_cat = data_baru[category_cols]

    # transform 
    data_num_scaled = scaler.transform(data_num)
    data_num_scaled = pd.DataFrame(data_num_scaled, columns=numeric_cols)

    data_cat_encoded = encoder.transform(data_cat)
    data_cat_encoded = pd.DataFrame(data_cat_encoded, columns=encoder.get_feature_names_out(category_cols))

    # feature_columns
    data_final = pd.concat([data_num_scaled, data_cat_encoded], axis=1)
    data_final = data_final[feature_columns]

    # predict
    prediksi = model.predict(data_final)[0]
    prediksi_proba = model.predict_proba(data_final)[0]

    # hasil
    st.subheader("Hasil Prediksi")

    warna = {"Podium": "🥇", "Poin": "🔟", "Non Poin": "❌"}
    st.markdown(f"### {warna.get(prediksi, '')} Kategori: **{prediksi}**")

    df_proba = pd.DataFrame({
        'Kategori': model.classes_,
        'Probabilitas': prediksi_proba
    }).sort_values('Probabilitas', ascending=False)

    st.bar_chart(df_proba.set_index('Kategori'))
    st.dataframe(df_proba, use_container_width=True, hide_index=True)

st.divider()
st.caption("Model: Decision Tree Classifier — Dataset: F1 Historical Results")
