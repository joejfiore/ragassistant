from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os

# Initialize FastAPI app
app = FastAPI()

# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Load ChromaDB vector store
persist_directory = "Chroma_DB/"
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# GPT-4 LLM setup
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4-turbo", temperature=0)

# Set up RetrievalQA chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=False
)

# Query endpoint
@app.post("/query")
async def query_rag(prompt: dict):
    query_text = prompt.get("query")
    response = retrieval_chain(query_text)
    return {"answer": response["result"]}
