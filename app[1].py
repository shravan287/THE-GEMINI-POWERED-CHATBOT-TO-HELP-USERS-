from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import google.generativeai as genai

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)

logger.info("Initializing OptimizedChatbot...")
logger.info("Loading Gemini model...")

# Set up Gemini model and vector retriever
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = Chroma(
    persist_directory="tax_vector_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

logger.info("Vector QA chain initialized successfully")

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_message = request.json.get("message")
        logger.info(f"[ask] Received: {user_message}")

        # Keyword-based routing
        tax_keywords = ["tax", "regime", "old regime", "new regime", "deduction", "income", "80c", "slab", "rebate"]
        if any(keyword in user_message.lower() for keyword in tax_keywords):
            result = qa_chain.run(user_message)
            return jsonify({"reply": result})

        # If no keywords match, fallback to general Gemini answer
        response = llm.generate_content(user_message)
        return jsonify({"reply": response.text})

    except Exception as e:
        logger.exception("Error processing request:")
        return jsonify({"error": str(e)}), 500

# Start Flask app
if __name__ == "__main__":
    logger.info("Starting optimized chatbot server...")
    port = int(os.environ.get("PORT", 5000))  # required for Render
    app.run(host="0.0.0.0", port=port)        # expose port to Render

