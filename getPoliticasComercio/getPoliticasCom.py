from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# =====================================================================
# Este programa usa el agente Ollama para interpretar un achivo txt
# y buscar respuestas a las preguntas formuladas en elñ prompt
# =====================================================================
# 1. Cargar el documento
with open("politicas.txt", "r", encoding="utf-8") as f:
    texto = f.read()

# print(f" Total caracteres: {len(texto)}")

# Crear documento
doc = Document(page_content=texto, metadata={"fuente": "politicas.txt"})

# Aumentar chunk_size para mantener contexto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ← Aumentado de 200 a 1000
    chunk_overlap=100     # ← Aumentado solapamiento
)
chunks = text_splitter.split_documents([doc])
# print(f" Fragmentos creados: {len(chunks)}")

# 2. Embeddings con llama3.2 (ya lo tienes)
embeddings = OllamaEmbeddings(model="llama3.2")

# 3. Vector store
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # ← Más documentos

# 4. LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# 5. Prompt mejorado
system_prompt = (
    "Eres un asistente de soporte técnico. Responde SOLO usando la información del contexto proporcionado.\n\n"
    "Si la respuesta está en el contexto, responde con la información exacta.\n"
    "Si NO encuentras la respuesta en el contexto, responde: 'La información no está disponible en el manual.'\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. RAG chain
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Probar con varias preguntas
preguntas = [
    "¿cual es el Canales Oficiales:?",
    "¿Cuántos días tengo para devolver un producto?",
    "¿Qué garantía tienen los productos tecnológicos?"
]

for pregunta in preguntas:
    print(f"\n{'='*50}")
    print(f"Pregunta: {pregunta}")
    respuesta = rag_chain.invoke(pregunta)
    print(f"Respuesta: {respuesta}")