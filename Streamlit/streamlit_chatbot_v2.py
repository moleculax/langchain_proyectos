import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv, find_dotenv

# Busca .env automáticamente en carpetas superiores
load_dotenv(find_dotenv())
# Se ejecuta: streamlit run streamlit_chatbot_v2.py en la terminal para iniciar la app
# USAMOS LA API KEY DE GEMINI
api_key = os.getenv("GOOGLE_API_KEY")
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
hide_deploy = """
    <style>
        .stAppDeployButton {display: none;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_deploy, unsafe_allow_html=True)




# Configuración inicial
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("Chatbot Básico con LangChain + Gemini")
st.markdown(
    "Este es un *chatbot  construido con LangChain + Gemini + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

# with st.sidebar:
#     st.header("Configuración")
#     temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
#     model_name = st.selectbox("Modelo", ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"])

temperature = 0.5
model_name = "gemini-2.5-flash"
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


        # Almacenamos el mensaje en la memoria de streamlit
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        # =====================================================
        respuesta = chat_model.invoke(st.session_state.mensajes)
        with st.chat_message("assistant"):
            st.markdown(respuesta.content)

            st.session_state.mensajes.append(AIMessage(content=respuesta.content))