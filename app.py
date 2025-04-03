import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno (local)
load_dotenv()

# Inicializar cliente de OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit config
st.set_page_config(page_title="Forecast App", layout="centered")
st.title("📈 Forecast con Chronos-Bolt + GPT-4o")
st.markdown("Subí tu CSV, explicá tu problema y dejá que la inteligencia artificial lo analice.")

# Subida de archivo CSV
uploaded_file = st.file_uploader("📂 Subí tu archivo CSV con fechas y valores", type=["csv"])

# Ingreso de contexto y objetivo
context = st.text_area("📝 Explicá el contexto del problema")
goal = st.text_area("🎯 ¿Qué te gustaría conocer o estimar?")

# Función para consultar Hugging Face Chronos-Bolt
def query_chronos(prompt):
    api_url = "https://api-inference.huggingface.co/models/amazon/chronos-bolt-base"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
    payload = {"inputs": prompt}

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        raise Exception(f"Error de Hugging Face: {response.status_code} - {response.text}")

# Acción principal
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Vista previa de los datos:")
    st.dataframe(df)

    if st.button("🚀 Analizar" and context and goal):
        with st.spinner("Generando prompt para Chronos con ChatGPT..."):
            try:
                # Paso 1: Crear prompt con ChatGPT
                user_input = f"""
                Tengo una serie temporal con estos datos:
                {df.head(10).to_string(index=False)}

                Este es el contexto del problema: {context}
                Este es el objetivo del usuario: {goal}

                Por favor, generá un prompt en inglés en el formato correcto para que pueda usarlo con Chronos-Bolt de Amazon. No expliques, solo devolvé el prompt listo para usar.
                """

                chat_prompt = [
                    {"role": "system", "content": "Sos un asistente experto en series temporales."},
                    {"role": "user", "content": user_input}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    temperature=0.3
                )

                chronos_prompt = response.choices[0].message.content

            except Exception as e:
                st.error("❌ Error al generar el prompt con ChatGPT")
                st.error(e)
                st.stop()

        with st.spinner("Obteniendo predicción de Chronos-Bolt..."):
            try:
                prediction = query_chronos(chronos_prompt)
            except Exception as e:
                st.error("❌ Error al consultar Chronos-Bolt")
                st.error(e)
                st.stop()

        st.subheader("📈 Predicción generada por Chronos-Bolt")
        st.code(prediction)

        with st.spinner("Generando informe explicativo con GPT-4o..."):
            try:
                final_prompt = f"""
                El usuario cargó esta serie temporal:
                {df.head(10).to_string(index=False)}

                Su objetivo era: {goal}
                Contexto del problema: {context}

                Esta fue la predicción generada por Chronos-Bolt:
                {prediction}

                Por favor, generá un informe explicativo en español, simple, para alguien que no es experto en datos.
                """

                explanation_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Sos un analista experto que redacta informes claros y simples."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.5
                )

                explanation = explanation_response.choices[0].message.content
                st.subheader("🧠 Informe generado con GPT-4o")
                st.write(explanation)

            except Exception as e:
                st.error("❌ Error al generar el informe con ChatGPT")
                st.error(e)
                st.stop()