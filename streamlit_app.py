import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el label encoder
modelo2 = joblib.load("modelo_rf_all.pkl")
le2 = joblib.load("label_encoder_all.pkl")

# Variables usadas en el modelo 2 (todas las variables)
todas_vars = list(modelo2.feature_names_in_)

# Cargar el dataset para obtener los valores min y max de cada variable
df = pd.read_csv("Crop_recommendation.csv")

# Si las variables están traducidas en el modelo, traducir columnas del dataset
traducciones_columnas = {
    'temperature': 'temperatura',
    'humidity': 'humedad',
    'rainfall': 'lluvia',
    'label': 'cultivo'
}
df = df.rename(columns={k: v for k, v in traducciones_columnas.items() if k in df.columns})

st.title("Recomendación de Cultivo (Modelo con todas las variables)")

inputs = []
for var in todas_vars:
    if var in df.columns:
        min_val = float(df[var].min())
        max_val = float(df[var].max())
        valor = st.number_input(
            f"Ingrese el valor de {var} (mín: {min_val:.2f}, máx: {max_val:.2f})",
            min_value=min_val, max_value=max_val, value=min_val
        )
    else:
        valor = st.number_input(f"Ingrese el valor de {var}", value=0.0)
    inputs.append(valor)

if st.button("Predecir cultivo recomendado"):
    df_nuevo = pd.DataFrame([inputs], columns=todas_vars)
    prediccion = modelo2.predict(df_nuevo)
    cultivo = le2.inverse_transform(prediccion)[0]
    st.success(f"Cultivo recomendado: {cultivo}")
