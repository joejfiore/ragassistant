import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load OpenAI API key securely from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Create and persist embeddings
persist_directory = "Chroma_DB/"
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

vectordb = Chroma.from_documents(
    documents=[],  # Replace with your document list
    embedding=embeddings,
    persist_directory=persist_directory
)

vectordb.persist()
print("Embeddings stored successfully.")
