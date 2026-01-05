import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Path to your PDF
pdf_path = r"C:\Users\Admin\OneDrive\Desktop\tax regime document\tax project document.pdf"


# Step 1: Extract text from PDF
print("📄 Extracting text from PDF...")
with pdfplumber.open(pdf_path) as pdf:
    full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Step 2: Split text into chunks
print("🔍 Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# Step 3: Embed using Gemini-compatible embeddings
print("🧠 Generating embeddings...")
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Step 4: Store in Chroma vector DB
print("💾 Saving to vector DB (tax_vector_db)...")
vectorstore = Chroma.from_documents(
    documents,
    embedding,
    persist_directory="tax_vector_db"
)

print("✅ Vector DB built successfully and stored in /tax_vector_db")
