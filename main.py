from fastapi import FastAPI
from pydantic import BaseModel
from qa_engine import ask_question
from fastapi.middleware.cors import CORSMiddleware
import some_chain_logic  # import your RetrievalQA setup

from fastapi.staticfiles import StaticFiles
import logging

logging.basicConfig(filename="queries.log", level=logging.INFO)

app = FastAPI()
# Allow requests from anywhere (for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with Netlify domain later
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "DocsBot is live!"}

@app.post("/ask")
def ask(query: Query):
    try:
        result = ask_question(query.question)
        logging.info(f"Q: {query.question} -> A: {result}")
        return result
    except Exception as e:
        logging.error(f"Error while answering: {query.question} -> {e}")
        return {"error": f"Something went wrong: {str(e)}"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render sets this
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

