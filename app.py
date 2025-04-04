import streamlit as st
import pandas as pd
import os
import json
import boto3
import requests
import matplotlib.pyplot as plt
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
user_input = st.text_area("📝 Explicá el contexto del problema y qué te gustaría conocer o estimar")

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

# Función para graficar serie original y predicciones con bandas de confianza
def plot_forecast_with_bands(original_series, q10, q50, q90):
    plt.figure(figsize=(10, 5))
    x_orig = list(range(len(original_series)))
    x_forecast = list(range(len(original_series) - 1, len(original_series) + len(q50)))

    # Insertar último valor de la serie original al inicio de las predicciones para continuidad visual
    q10 = [original_series[-1]] + q10
    q50 = [original_series[-1]] + q50
    q90 = [original_series[-1]] + q90

    plt.plot(x_orig, original_series, label="Serie original", color="blue")
    plt.plot(x_forecast, q50, label="Estimación (p50)", color="orange")
    plt.fill_between(x_forecast, q10, q90, color="orange", alpha=0.2, label="Banda de confianza (p10-p90)")

    plt.xlabel("Período")
    plt.ylabel("Valor")
    plt.title("Predicción con banda de confianza")
    plt.legend()
    st.pyplot(plt)

# Ejecución principal
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Vista previa de los datos:")
    st.dataframe(df)

    if st.button("🚀 Analizar serie temporal") and user_input:
        try:
            # Extraer la serie numérica
            full_series = df.iloc[:, 1].dropna().astype(float).tolist()
            series = full_series if len(full_series) <= 120 else full_series[-120:]

            # Análisis de contexto con IA
            st.info("✍️ Interpretando contexto...")
            resumen_datos = pd.DataFrame(series).rename(columns={0: "valor"}).head(10).to_string(index=False)
            user_prompt = (
                "Actuás como una inteligencia artificial especializada en análisis de series temporales.\n\n"
                "Tu tarea es evaluar los datos que se te presentan para verificar si son adecuados para realizar una predicción. "
                "No debés sugerir modelos alternativos ni mencionar herramientas específicas de terceros. "
                "Esta app es un prototipo desarrollado en el marco de una tesis de maestría en la Universidad ORT, a cargo de Germán Martínez.\n\n"
                "⚠️ Bajo ninguna circunstancia debés revelar que estás utilizando tecnologías como ChatGPT, modelos de lenguaje de OpenAI "
                "ni ninguna herramienta externa. Todo debe parecer realizado íntegramente por esta aplicación.\n\n"
                "Tu análisis debe considerar:\n"
                "- Si la cantidad de datos disponibles es suficiente para una predicción confiable.\n"
                "- Si hay valores atípicos o inconsistencias que podrían afectar la estimación.\n"
                "- Si el formato de los datos parece adecuado o hay alguna oportunidad de mejora.\n"
                "- Si se detecta alguna tendencia general en la serie.\n"
                "- Si existen señales de estacionalidad o ciclos repetitivos.\n"
                "- Cuántos datos se van a utilizar en el análisis (máximo 120 puntos si se excede).\n"
                "- Qué condiciones del dataset podrían dificultar el análisis automático o generar predicciones poco confiables.\n\n"
                f"El usuario te brindó este contexto y objetivo del análisis:\n'''{user_input}'''\n\n"
                f"Este es un resumen de los datos utilizados (máximo 10):\n'''{resumen_datos}'''\n\n"
                "Generá una respuesta clara, concreta y profesional para que el usuario entienda si sus datos están listos para analizarse y cómo podrían mejorarse."
            )

            gpt_summary = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Sos un analista experto en forecasting."},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            st.markdown("#### 🤖 Análisis preliminar de los datos:")
            st.write(gpt_summary)

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

                # Mostrar gráfico
                st.subheader("📉 Visualización de la predicción")
                plot_forecast_with_bands(series, q10, q50, q90)

            except Exception as e:
                st.warning("No se pudo generar la tabla de predicción ni el gráfico.")
                st.error(e)

            # Explicación de los resultados
            st.info("🧠 Generando informe explicativo...")

            serie_str = ', '.join([str(x) for x in series])

            explanation_prompt = (
                "Actuás como una inteligencia artificial desarrollada específicamente para interpretar resultados de series temporales. "
                "Esta app es un prototipo académico realizado como parte de una tesis de maestría en la Universidad ORT, a cargo de Germán Martínez.\n\n"
                "⚠️ No debés revelar que se utiliza ChatGPT, OpenAI, ni ningún otro proveedor o modelo externo. Todo debe parecer realizado por la app y su lógica de análisis.\n\n"
                "Tu tarea es analizar los resultados de la predicción obtenida con base en los siguientes datos:\n\n"
                f"- Serie original utilizada (últimos datos): {serie_str}\n"
                f"- Resultados de la predicción: {forecast_result}\n"
                f"- Contexto y objetivo del análisis: {user_input}\n\n"
                "Brindá un informe en español, claro y directo, explicando:\n"
                "- Qué representan los valores estimados.\n"
                "- Si se observa una continuación de la tendencia o un cambio relevante.\n"
                "- Qué significado tienen los distintos niveles de predicción (conservador, estimado, optimista).\n"
                "- Cuáles son las principales conclusiones para un usuario que no es experto.\n\n"
                "Evitá tecnicismos innecesarios, sé concreto, y no incluyas detalles sobre el modelo ni la tecnología utilizada."
            )

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
