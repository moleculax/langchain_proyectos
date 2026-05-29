# hello_word.py - Con API key explícita
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.7)

pregunta = "¿En que año llego el ser humano a la luna por primera vez?"
print("Pregunta: ", pregunta)
respuesta = llm.invoke(pregunta)
print("respuesta del modelo: ", respuesta.content)