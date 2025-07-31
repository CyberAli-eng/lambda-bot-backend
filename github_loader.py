import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ⚠️ Best Practice: Use env variable instead of hardcoding in production
GITHUB_TOKEN = "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"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

REPO = "CyberAli-eng/lambda-docs-bot"
BRANCH = "main"

def fetch_github_docs():
    print("[INFO] Fetching file list from GitHub API...")
    tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"

    resp = requests.get(tree_url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    file_urls = [
        f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{item['path']}"
        for item in data.get("tree", [])
        if item["path"].endswith(".txt") and "data_batches" in item["path"]
    ]

    print(f"[INFO] Found {len(file_urls)} .txt files.")

    docs = []
    for url in file_urls:
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            docs.append(r.text)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")

    print(f"[INFO] Fetched {len(docs)} documents.")
    return docs

def split_docs(raw_texts):
    print("[INFO] Splitting documents...")
    docs = [Document(page_content=txt) for txt in raw_texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks

