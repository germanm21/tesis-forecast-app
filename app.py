import streamlit as st
import pandas as pd
import time
import openai
import os
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


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
            forecast_result = {
                "item": "Ventas",
                "predicción próximos 5 días": [102, 108, 115, 112, 119]
            }

        st.success("✅ Análisis completado.")
        st.subheader("🔮 Resultados de la predicción")
        st.json(forecast_result)

        st.subheader("🧠 Explicación generada con ChatGPT")

        # Armamos el prompt
        prompt = f"""
        Estos son los resultados de una predicción de series temporales: {forecast_result}
        El usuario proporcionó esta descripción del contexto: "{description}"

        Explicá en lenguaje natural qué significan los resultados, si hay alguna tendencia o patrón, y qué puede concluir una persona que no sabe de ciencia de datos.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # o "gpt-3.5-turbo" si tenés acceso limitado
                messages=[
                    {"role": "system", "content": "Sos un analista experto en series temporales."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            explanation = response["choices"][0]["message"]["content"]
            st.write(explanation)

        except Exception as e:
            st.error("Error al conectarse con OpenAI:")
            st.error(e)