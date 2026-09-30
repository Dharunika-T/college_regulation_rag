from pathlib import Path

from pypdf import PdfReader

from src.rag import get_collection


DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"


def split_text(text, size=1500):
    text = " ".join(text.split())
    step = size - 150
    return [text[i : i + size] for i in range(0, len(text), step)]


def index_pdfs():
    collection = get_collection()
    counts = {}

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        documents, metadata, ids = [], [], []

        for page_number, page in enumerate(reader.pages, start=1):
            for chunk_number, chunk in enumerate(split_text(page.extract_text() or "")):
                documents.append(chunk)
                metadata.append({"source": pdf_path.name, "page": page_number})
                ids.append(f"{pdf_path.name}:{page_number}:{chunk_number}")

        collection.delete(where={"source": pdf_path.name})
        if documents:
            collection.upsert(ids=ids, documents=documents, metadatas=metadata)
        counts[pdf_path.name] = len(documents)

    if not counts:
        raise FileNotFoundError(f"No PDF files found in {DOCUMENTS_DIR}.")

    return counts
