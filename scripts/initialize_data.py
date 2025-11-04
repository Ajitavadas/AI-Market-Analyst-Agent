from src.vector_store import initialize_document_store
with open("data/innovate_inc_report.txt", "r") as f:
    document_text = f.read()
vector_store = initialize_document_store(document_text)
print("Vector store initialized successfully")