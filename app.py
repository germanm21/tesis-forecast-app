import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Forecast App", layout="centered")

st.title("📈 Forecast con Amazon Chronos + GPT")
st.markdown("Subí tu CSV, explicá qué significan los datos, y obtené un análisis automático.")

uploaded_file = st.file_uploader("📂 Subí tu archivo CSV", type=["csv"])
description = st.text_area("📝 Explicación o contexto de los datos (opcional)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Vista previa de los datos:")
    st.dataframe(df)

    if st.button("🚀 Analizar con Chronos + GPT (simulado)"):
        with st.spinner("Procesando datos con Chronos..."):
            time.sleep(2)  # Simula tiempo de procesamiento
            # Simulamos una predicción
            forecast_result = {
                "item": "Ventas",
                "predicción próximos 5 días": [102, 108, 115, 112, 119]
            }

        st.success("✅ Análisis completado.")
        st.subheader("🔮 Resultados de la predicción")
        st.json(forecast_result)

        st.subheader("🧠 Explicación generada")
        explanation = f"""
        Según los datos proporcionados, se realizó una predicción de las ventas para los próximos 5 días.
        Se observa una tendencia levemente creciente. Esto podría estar relacionado con un aumento estacional
        o una campaña de marketing reciente.

        Recordá que esta es una simulación. Pronto podrás conectarte con Amazon Chronos y ChatGPT de verdad.
        """
        st.write(explanation)
