from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest")
response = llm.invoke("Dime un dato curioso sobre la programación")
print(response)
