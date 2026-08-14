import os
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()

# Make sure GOOGLE_API_KEY exists
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your .env file."
    )

# 2. IMPORTS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model


# 3. LOAD PDF

pdf_path = "data/AI.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))

# 4. SPLIT PDF INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 5. EMBEDDING MODEL

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# 6. VECTOR DATABASE

persist_dir = "chroma_db"


if os.path.exists(persist_dir) and os.listdir(persist_dir):

    # Load existing database
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    print("Loaded existing Chroma database.")

else:

    # Create database for the first time
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("Created new Chroma database.")

# 7. RETRIEVER
# The retriever searches Chroma for chunks relevant to the
# user's question.
# k=3 means retrieve the 3 most relevant chunks.

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6
    }
)

# 8. GEMINI CHAT MODEL
# - responsible for generating the final answer.
# GOOGLE_API_KEY is read automatically from the environment.

model = init_chat_model(
    model="google_genai:gemini-3.5-flash"
)

print("Gemini chat model loaded.")


# 9. ASK USER A QUESTION

question = input("\nAsk a question about the PDF: ")


# 10. RETRIEVE RELEVANT DOCUMENTS

print("\nSearching the PDF...")

retrieved_docs = retriever.invoke(question)

# 11. SHOW RETRIEVED DOCUMENTS

print("\n--- RETRIEVED DOCUMENTS ---")

for i, doc in enumerate(retrieved_docs):

    print(f"\n### RESULT {i + 1}")

    print("Page:", doc.metadata.get("page"))

    print("Source:", doc.metadata.get("source"))

    print("\nContent:")

    print(doc.page_content)

# 12. JOIN CONTEXT

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

# 13. BUILD PROMPT FOR GEMINI

prompt = f"""
You are a helpful assistant answering questions about the
research paper "Speeding up to keep up: exploring the use of
AI in the research process".

Your job is to answer the user's question using ONLY the
information contained in the retrieved context.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Answer the question directly.
4. Explain the answer clearly.
5. Give 2 to 5 sentences when possible.
6. If the context does not contain enough information,
   say exactly:

   "The answer is not available in the provided context."

Retrieved context:
----------------------------

{context}

----------------------------

Question:
{question}

Answer:
"""

# 14. SEND PROMPT TO GEMINI

print("\nGenerating answer...")

response = model.invoke(prompt)

# Get only the actual answer text
if isinstance(response.content, list):
    answer = "".join(
        block["text"]
        for block in response.content
        if isinstance(block, dict) and "text" in block
    )
else:
    answer = response.content

print("\n==============================")
print("Question:")
print(question)

print("\nAnswer:")
print(answer)

print("==============================")

