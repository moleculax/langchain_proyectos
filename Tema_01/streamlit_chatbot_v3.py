import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv, find_dotenv

# ❌ ERROR LÍNEA 9: Import incorrecto - estas variables no existen en ese módulo
# from Tema_01.streamlit_chatbot import response_placeholder, full_response
# ✅ CORRECCIÓN: Eliminar esta línea

# Busca .env automáticamente en carpetas superiores
load_dotenv(find_dotenv())
# Se ejecuta: streamlit run streamlit_chatbot_v2.py en la terminal para iniciar la app
# USAMOS LA API KEY DE GEMINI LA CUAL ESTA COMO VARIABLE DE ENTORNO
api_key = os.getenv("GOOGLE_API_KEY")
# OLLAMA3.2 NO NECESITA API KEY PORQUE ESTA INSTALADO COMO AGENTEB IA LOCALMENTE
# Configuración inicial - CON BOTÓN DEPLOY OCULTO
st.set_page_config(
    page_title="Chatbot Básico",
    page_icon="🤖",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Ocultar botón Deploy
# header {visibility: hidden;}
hide_deploy = """
    <style>
        .stAppDeployButton {display: none;}

        footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_deploy, unsafe_allow_html=True)

# Configuración inicial
st.title("Chatbot Básico con LangChain + Gemini + Ollama3.2")
st.markdown(
    "Este es un *chatbot  construido con LangChain + Gemini, Ollama + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")
with st.sidebar:
    st.header("Configuración de modelo y temperatura")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelos disponible", [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "ollama-llama3.2",
        "ollama-mistral",
        "ollama-phi3"
    ], index=3)  # Por defecto (índice 3)


# Seleccionar el modelo según la opción elegida
if model_name.startswith("ollama-"):
    ollama_model = model_name.replace("ollama-", "")
    chat_model = ChatOllama(model=ollama_model, temperature=temperature)
else:
    chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

# Inicializar el historial de mensajes en session_state
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Crear el template de prompt con comportamiento específico
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres un asistente útil y amigable llamado ChatBot Pro. 

Historial de conversación:
{historial}

Responde de manera clara y concisa a la siguiente pregunta: {mensaje}"""
)

# Crear cadena usando LCEL (LangChain Expression Language)
cadena = prompt_template | chat_model

# Renderizar historial existente,
# muestra el mensaje previo en lña pantalla
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # no mostrar mensajes del sistema al usuario por pantalla

    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

if st.button("🗑️ Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()

# Input de usuario
# Cuadro de entrada de texto del usuario

pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Mostramos por pantalla el mensaje del usuario en la interfaz
    # y almacenar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in cadena.stream({"mensaje": pregunta, "historial": st.session_state.mensajes}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        # Almacenamos el mensaje en la memoria de streamlit
        st.session_state.mensajes.append(HumanMessage(content=pregunta))

        st.session_state.mensajes.append(AIMessage(content=full_response))


    except Exception as e:
        st.error(f"Error: {str(e)}")