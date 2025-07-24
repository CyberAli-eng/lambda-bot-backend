from langchain.chains import RetrievalQA
# from langchain_chroma import Chroma
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
import subprocess

# === OpenRouter API Setup ===
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

DB_DIR = "./chroma_db"
COLLECTION_NAME = "lambdatest_docs"

# === Regenerate ChromaDB if missing ===
if not os.path.exists(DB_DIR):
    print("⚙️ No ChromaDB found, generating it using embed_docs.py...")
    subprocess.run(["python", "embed_docs.py"], check=True)
    print("✅ ChromaDB generated.")

# === Load DB ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding,
    collection_name=COLLECTION_NAME
)

# === LLM & Retriever ===
llm = ChatOpenAI(temperature=0, model_name="mistralai/mistral-7b-instruct")
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === Main QA Function ===
def ask_question(query: str) -> dict:
    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        references = []

        for doc in result["source_documents"]:
            matches = re.findall(r"https://www\.lambdatest\.com/support/[^\s\"']+", doc.page_content)
            for url in matches:
                if url not in references:
                    references.append(url)

        return {"answer": answer, "references": references or [""]}
    except Exception as e:
        return {"answer": "Sorry, something went wrong.", "references": [str(e)]}
