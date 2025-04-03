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

ENDPOINT_NAME = "jumpstart-dft-autogluon-forecasting-20250403-120703"

# Streamlit config
st.set_page_config(page_title="Forecast App", layout="centered")
st.title("📈 Smart Forecast")
st.markdown("Subí tu CSV, explicá tu problema y dejá que la inteligencia artificial lo analice.")

# Cargar archivo CSV
uploaded_file = st.file_uploader("📂 Subí tu archivo CSV con fechas y valores", type=["csv"])
context = st.text_area("📝 Explicá el contexto del problema")
goal = st.text_area("🎯 ¿Qué te gustaría conocer o estimar?")

# Nuevo slider para seleccionar prediction_length
prediction_length = st.slider(
    "🔢 ¿Cuántos períodos querés predecir?",
    min_value=1,
    max_value=30,
    value=5,
    help="Elegí la cantidad de períodos futuros que querés estimar."
)

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
            st.markdown("#### 🤖 GPT-4o interpreta el contexto:")
            st.write(gpt_summary)
        
            # Extraer la serie numérica
            series = df.iloc[:, 1].dropna().astype(float).tolist()

            # Predecir con Chronos desde SageMaker usando el valor elegido
            st.info("🔮 Prediciendo valores futuros...")
            forecast_result = predict_with_sagemaker(series, prediction_length=prediction_length)

            # Mostrar tabla legible (ocultamos el diccionario crudo)
            try:
                pred = forecast_result["predictions"][0]
                q10 = pred.get("0.1", [])
                q50 = pred.get("0.5", [])
                q90 = pred.get("0.9", [])

                df_pred = pd.DataFrame({
                    "Día": list(range(1, len(q50)+1)),
                    "Criterio conservador (p10)": q10,
                    "Estimación (p50)": q50,
                    "Criterio optimista (p90)": q90
                })

                st.subheader("📈 Predicción")
                st.dataframe(df_pred, use_container_width=True)

            except Exception as e:
                st.warning("No se pudo generar la tabla de predicción.")
                st.error(e)

            # Explicación de los resultados
            st.info("🧠 Generando informe explicativo...")

            serie_para_prompt = series if len(series) <= 120 else series[-120:]
            serie_str = ', '.join([str(x) for x in serie_para_prompt])

            explanation_prompt = f"""
            Se hizo una predicción de series temporales con estos datos:
            Serie original: {serie_str}
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

            st.subheader("🧾 Informe final")
            st.write(explanation)

        except Exception as e:
            st.error("❌ Ocurrió un error en el análisis")
            st.error(e)
