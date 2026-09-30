from pathlib import Path

import chromadb


CHROMA_PATH = Path(__file__).resolve().parents[1] / "chroma_db"


def get_collection():
	client = chromadb.PersistentClient(path=str(CHROMA_PATH))
	return client.get_or_create_collection(name="legal_documents")


def search_documents(query, n_results=6, source=None):
	collection = get_collection()
	if collection.count() == 0:
		return []

	options = {
		"query_texts": [query],
		"n_results": min(n_results, collection.count()),
	}
	if source:
		options["where"] = {"source": source}
	results = collection.query(**options)
	return [
		{"text": text, **metadata}
		for text, metadata in zip(results["documents"][0], results["metadatas"][0])
	]
