from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from src.helper import download_embeddings
import os

app = Flask(__name__)


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_embeddings()
index_name = "medical-chatbot"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful medical assistant.
Use the following context to answer the user's question.
If you do not know the answer, say that you do not know.
Use a maximum of three sentences, and keep the answer medically accurate, clear, and concise.

Context:
{context}

Question: {question}
Answer:
"""
)


def create_chain():
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
        verbose=False
    )

    return qa_chain, memory


qa_chain, memory = create_chain()

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User: {msg}")

    result = qa_chain.invoke({"question": msg})
    answer = result["answer"]
    print(f"AI: {answer}")

    return str(answer)


@app.route("/reset", methods=["POST"])
def reset_memory():
    global qa_chain, memory
    qa_chain, memory = create_chain()
    print("Memory reset successfully.")
    return jsonify({"status": "success", "message": "Memory cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
