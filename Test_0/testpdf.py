# testpdf.py - Usando solo llama3.2
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import pypdf

# Usar llama3.2 para todo
EMBEDDING_MODEL = "llama3.2"
CHAT_MODEL = "llama3.2"

def leer_pdf(ruta):
    """Extrae texto de un PDF usando pypdf"""
    texto_completo = ""
    try:
        with open(ruta, 'rb') as archivo:
            lector = pypdf.PdfReader(archivo)
            for num_pagina, pagina in enumerate(lector.pages, 1):
                texto = pagina.extract_text()
                if texto:
                    texto_completo += f"\n--- Página {num_pagina} ---\n{texto}"
        return texto_completo
    except Exception as e:
        print(f"Error al leer PDF: {e}")
        return None

def main():
    PDF_PATH = "documento.pdf"  # Cambia por tu PDF
    
    print("📄 Cargando PDF...")
    texto = leer_pdf(PDF_PATH)
    
    if not texto:
        print("No se pudo leer el PDF. Verifica la ruta.")
        return
    
    print(f"✅ Texto extraído: {len(texto)} caracteres")
    
    # Crear documento y dividir en chunks
    print("✂️  Dividiendo en fragmentos...")
    doc = Document(page_content=texto, metadata={"fuente": PDF_PATH})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents([doc])
    print(f"📑 {len(chunks)} fragmentos creados")
    
    # Crear embeddings y vector store
    print("🧠 Generando embeddings con llama3.2...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)
    print("✅ Vector store listo")
    
    # Configurar modelo de chat
    llm = ChatOllama(model=CHAT_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("\n🤖 ¡Listo! Puedes hacer preguntas sobre el PDF.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        pregunta = input("❓ Pregunta: ").strip()
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("👋 Hasta luego!")
            break
        
        if not pregunta:
            continue
        
        print("🔍 Buscando respuesta...")
        docs = retriever.invoke(pregunta)
        contexto = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto. 
Si la respuesta no está en el contexto, di 'No encontré información sobre eso en el documento'.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""
        
        respuesta = llm.invoke(prompt)
        print(f"\n💬 Respuesta: {respuesta.content}\n")

if __name__ == "__main__":
    main()