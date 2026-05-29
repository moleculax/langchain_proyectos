from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key="tuapikey"
)

pregunta = "¿En que año llego el ser humano a la luna por primera vez y en cual mision?"
print("Pregunta: ", pregunta)
respuesta = llm.invoke(pregunta)
print("respuesta del modelo: ", respuesta.content)