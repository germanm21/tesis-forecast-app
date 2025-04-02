import os
from openai import OpenAI

# Usamos la variable de entorno con tu API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Listamos todos los modelos disponibles en tu cuenta
models = client.models.list()

for model in models.data:
    print(model.id)
