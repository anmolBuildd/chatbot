from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

loader = TextLoader("data/output.txt")  # your full transcript here
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory="db")
vectordb.persist()

print("✅ Seminar transcript ingested")