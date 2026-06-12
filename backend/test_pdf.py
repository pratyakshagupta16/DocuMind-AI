from dotenv import load_dotenv
load_dotenv()

from utils.pdf_loader import load_pdf
from utils.embedder import split_documents
from utils.vector_store import get_embeddings, create_vector_store
from utils.generator import generate_answer

# STEP 1: Extract text and metadata as Documents
print("Extracting PDF contents...")
docs = load_pdf("data/sample.pdf")
print(f"Total pages: {len(docs)}")

# STEP 2: Split into chunks
print("Splitting documents into chunks...")
chunks = split_documents(docs)
print(f"Total chunks: {len(chunks)}")

# STEP 3: Create vector DB
print("Initializing embeddings and creating vector store...")
embeddings = get_embeddings()
vector_db = create_vector_store(chunks, embeddings)

# STEP 4: Ask query
query = "What skills does the person have?"
print(f"Searching for query: '{query}'")
search_docs = vector_db.similarity_search(query, k=3)

# STEP 5: Generate answer
print("Generating final answer...")
answer = generate_answer(query, search_docs)

print("\n--- FINAL ANSWER ---\n")
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
print(answer)