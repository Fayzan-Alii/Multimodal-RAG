import chromadb
from chromadb.utils import embedding_functions

# Update to the current ChromaDB client initialization
client = chromadb.PersistentClient(path='data/chroma')
collection = client.get_or_create_collection(
    'multimodal',
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='clip-ViT-B-32'
    )
)

def add_texts(chunks, embs):
    ids = [c['chunk_id'] for c in chunks]
    docs= [c['content'] for c in chunks]
    metas = [{k:c[k] for k in('source','page','type')} for c in chunks]
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs.tolist())
    # No need to call persist() with PersistentClient

def add_images(image_ids, embs, metas):
    # Create meaningful descriptions for each image
    documents = [f"Image from document '{meta.get('source', 'unknown')}', page {meta.get('page', 'unknown')}" for meta in metas]
    collection.add(ids=image_ids, documents=documents, metadatas=metas, embeddings=embs.tolist())