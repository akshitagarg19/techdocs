import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Please set OPENAI_API_KEY in .env")

# 1. Load TypeScript Book markdown files
print("Loading TypeScript Book markdown files...")
loader = DirectoryLoader("typescript-book", glob="**/*.md")
raw_docs = loader.load()

# 2. Split docs into smaller chunks
print(f"Loaded {len(raw_docs)} documents, splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(raw_docs)
print(f"Split into {len(docs)} chunks.")

# 3. Create embeddings and vectorstore
print("Creating embeddings and vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# 4. Setup Retrieval QA chain
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

@app.get("/search")
def search(q: str = Query(..., description="Developer question")):
    """Search API for developer questions."""
    result = qa_chain(q)
    sources = list(set(doc.metadata.get("source", "unknown") for doc in result['source_documents']))
    return {
        "answer": result["result"],
        "sources": sources
    }
