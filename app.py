import streamlit as st
import pandas as pd
import os
import json
import boto3
import requests
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

ENDPOINT_NAME = "jumpstart-dft-autogluon-forecasting-20250403-032604"

# Streamlit config
st.set_page_config(page_title="Forecast App", layout="centered")
st.title("📈 Forecast con Chronos (SageMaker) + GPT")
st.markdown("Subí tu CSV, explicá tu problema y dejá que la inteligencia artificial lo analice.")

# Cargar archivo CSV
uploaded_file = st.file_uploader("📂 Subí tu archivo CSV con fechas y valores", type=["csv"])
context = st.text_area("📝 Explicá el contexto del problema")
goal = st.text_area("🎯 ¿Qué te gustaría conocer o estimar?")

# Función para predecir desde SageMaker

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

# Ejecución principal
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Vista previa de los datos:")
    st.dataframe(df)

    if st.button("🚀 Analizar serie temporal") and context and goal:
        try:
            # Usamos ChatGPT para interpretar el objetivo y generar contexto
            st.info("✍️ Interpretando contexto con GPT-4o...")
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
            st.markdown("#### 🤖 GPT-4o interpreta el contexto:")
            st.write(gpt_summary)
        
            # Extraer la serie numérica
            series = df.iloc[:, 1].dropna().astype(float).tolist()

            # Predecir con Chronos desde SageMaker
            st.info("🔮 Prediciendo con Chronos (SageMaker)...")
            forecast_result = predict_with_sagemaker(series, prediction_length=5)

            st.subheader("📈 Predicción de Chronos")
            st.write(forecast_result)

            # Explicación de los resultados
            st.info("🧠 Generando informe explicativo con GPT-4o...")
            explanation_prompt = f"""
            Se hizo una predicción de series temporales con estos datos:
            Serie original: {series[-10:]}
            Predicción: {forecast_result}

            Contexto: {context}
            Objetivo del usuario: {goal}

            Generá un informe simple y claro en español para alguien no experto.
            """

            explanation = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Sos un analista que redacta informes claros y concisos para negocio."},
                    {"role": "user", "content": explanation_prompt}
                ]
            ).choices[0].message.content

            st.subheader("🧾 Informe final generado con GPT-4o")
            st.write(explanation)

        except Exception as e:
            st.error("❌ Ocurrió un error en el análisis")
            st.error(e)