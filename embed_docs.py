import os
import shutil
from github_loader import fetch_github_docs, split_docs
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Config ===
DB_DIR = "./chroma_db"

# === Clean up old DB (if exists) ===
if os.path.exists(DB_DIR):
    print("[INFO] Removing old Chroma DB...")
    shutil.rmtree(DB_DIR)
os.makedirs(DB_DIR, exist_ok=True)

# === Load & split docs ===
print("[INFO] Fetching documents from GitHub...")
docs = fetch_github_docs()

if not docs:
    print("[❌ ERROR] No documents fetched. Exiting.")
    exit(1)

print(f"[INFO] Fetched {len(docs)} documents.")
print("[INFO] Splitting documents into chunks...")
chunks = split_docs(docs)
print(f"[INFO] Split into {len(chunks)} chunks.")

# === Embedding & Store ===
print("[INFO] Creating HuggingFace embeddings...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("[INFO] Creating Chroma vector store...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=DB_DIR,
    collection_name="lambdatest_docs"  # 👈 MUST match qa_engine.py
)
vectordb.persist()

print("[✅ SUCCESS] Vector DB created and saved to:", DB_DIR)

