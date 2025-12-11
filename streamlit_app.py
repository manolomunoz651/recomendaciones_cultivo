import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el label encoder
modelo2 = joblib.load("../modelos/modelo_rf_all.pkl")
le2 = joblib.load("../modelos/label_encoder_all.pkl")

# Variables usadas en el modelo 2 (todas las variables)
todas_vars = list(modelo2.feature_names_in_)

st.title("Recomendación de Cultivo (Modelo con todas las variables)")

inputs = []
for var in todas_vars:
    valor = st.number_input(f"Ingrese el valor de {var}", value=0.0)
    inputs.append(valor)

if st.button("Predecir cultivo recomendado"):
    df_nuevo = pd.DataFrame([inputs], columns=todas_vars)
    prediccion = modelo2.predict(df_nuevo)
    cultivo = le2.inverse_transform(prediccion)[0]
    st.success(f"Cultivo recomendado: {cultivo}")
