import streamlit as st
import pandas as pd
import os
import json
import boto3
import requests
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configurar clientes de OpenAI y AWS SageMaker
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

ENDPOINT_NAME = "jumpstart-dft-autogluon-forecasting-20250403-120703"

# Streamlit config
st.set_page_config(page_title="Forecast App", layout="centered")
st.title("📈 Smart Forecast")
st.markdown("Subí tu CSV, explicá tu problema y dejá que la inteligencia artificial lo analice.")

# Cargar archivo CSV
uploaded_file = st.file_uploader("📂 Subí tu archivo CSV con fechas y valores", type=["csv"])
context = st.text_area("📝 Explicá el contexto del problema")
goal = st.text_area("🎯 ¿Qué te gustaría conocer o estimar?")
prediction_length = st.slider("🔢 ¿Cuántos períodos querés predecir?", min_value=1, max_value=30, value=5)

# Función para predecir desde SageMaker (modo simple, sin configuración extra)
def predict_with_sagemaker(values, prediction_length=5):
    payload = {
        "inputs": [{"target": values}],
        "parameters": {"prediction_length": prediction_length}
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    return json.loads(response["Body"].read())

# Función para graficar la predicción simple

def plot_forecast(series, forecast):
    forecast = np.array(forecast)
    x_orig = list(range(len(series)))
    x_pred = list(range(len(series), len(series) + len(forecast)))

    plt.figure(figsize=(10, 5))
    plt.plot(x_orig, series, label="Serie original", color="blue")
    plt.plot(x_pred, forecast, label="Predicción", color="orange")
    plt.legend()
    plt.xlabel("Período")
    plt.ylabel("Valor")
    plt.title("Predicción de la serie temporal")
    st.pyplot(plt)

# Ejecución principal
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Vista previa de los datos:")
    st.dataframe(df)

    if st.button("🚀 Analizar serie temporal") and context and goal:
        try:
            # Interpretar el contexto con IA
            st.info("✍️ Interpretando contexto...")
            user_prompt = f"""
            El usuario subió esta serie temporal:
            {df.head(10).to_string(index=False)}

            Contexto: {context}
            Objetivo: {goal}

            ¿Podrías confirmar si los datos parecen válidos y sugerir qué podríamos predecir?
            """
            gpt_summary = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Sos un analista experto en forecasting."},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            st.markdown("#### 🤖 Interpretación del modelo:")
            st.write(gpt_summary)

            # Extraer la serie numérica
            series = df.iloc[:, 1].dropna().astype(float).tolist()

            # Predecir desde SageMaker
            st.info("🔮 Prediciendo valores futuros...")
            forecast_result = predict_with_sagemaker(series, prediction_length=prediction_length)

            # Validar formato del resultado
            forecast_values = forecast_result[0] if isinstance(forecast_result, list) else forecast_result.get("mean", [])

            if isinstance(forecast_values, dict):
                forecast_values = list(forecast_values.values())

            if not forecast_values:
                st.warning("⚠️ No se encontraron valores numéricos en la predicción para graficar.")
            else:
                st.subheader("📈 Predicción")
                st.write(forecast_values)

                st.subheader("📊 Visualización")
                plot_forecast(series, forecast_values)

                # Generar informe explicativo
                st.info("🧠 Generando informe explicativo...")
                serie_para_prompt = series if len(series) <= 120 else series[-120:]
                serie_str = ', '.join([str(x) for x in serie_para_prompt])

                explanation_prompt = f"""
                Se hizo una predicción de series temporales con estos datos:
                Serie original: {serie_str}
                Predicción: {', '.join([str(x) for x in forecast_values])}

                Contexto: {context}
                Objetivo del usuario: {goal}

                Generá un informe simple y claro en español para alguien no experto.
                Indicá si hay tendencias, estacionalidad o anomalías.
                """

                explanation = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Sos un analista que redacta informes claros y concisos para negocio."},
                        {"role": "user", "content": explanation_prompt}
                    ]
                ).choices[0].message.content

                st.subheader("🧾 Informe final")
                st.write(explanation)

        except Exception as e:
            st.error("❌ Ocurrió un error en el análisis")
            st.error(e)