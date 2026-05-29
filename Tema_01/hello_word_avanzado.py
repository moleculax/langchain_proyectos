from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
   api_key="tuapikey22"

)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Hola\n {nombre}, ¿Cómo estás? \n Asistente:"
)

# usar pipe en lugar de LLMChain
chain = plantilla | chat

resultado = chain.invoke({"nombre": "Pepe"})

#  resultado es un objeto, no string
print(resultado.content)