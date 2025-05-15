import time
from vectorstore import collection
from embedding import embed_texts, embed_images

# Cache for storing embeddings
embedding_cache = {}

def get_text_embedding(text):
    # Returns the embedding for a single text string
    return embed_texts([text])[0]

def get_image_embedding(image_path):
    # Returns the embedding for a single image file
    return embed_images([image_path])[0]

def retrieve_text(query, top_k=5, timeout=10):
    """Retrieve relevant text based on query with timeout"""
    try:
        # Check if query embedding is already in cache
        if query in embedding_cache:
            embedding = embedding_cache[query]
        else:
            # Start timer
            start_time = time.time()
            embedding = get_text_embedding(query)
            embedding_cache[query] = embedding
            
            # Check if embedding took too long
            if time.time() - start_time > timeout * 0.5:  # Allow half the timeout for embedding
                print(f"Warning: Embedding took {time.time() - start_time:.2f} seconds")
        
        # Second timer for retrieval
        start_time = time.time()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        if time.time() - start_time > timeout * 0.5:  # Allow half the timeout for retrieval
            print(f"Warning: Retrieval took {time.time() - start_time:.2f} seconds")
        
        hits = []
        if results and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                id = results['ids'][0][i]
                hit = {
                    'content': doc,
                    'meta': meta,
                    'chunk_id': id
                }
                hits.append(hit)
        return hits
    except Exception as e:
        print(f"Error in retrieve_text: {str(e)}")
        return []

def retrieve_image(image_path, top_k=5):
    """Retrieve relevant text based on image query"""
    try:
        # Generate embedding for the image
        img_embedding = get_image_embedding(image_path)
        
        # Query collection with the image embedding
        results = collection.query(
            query_embeddings=[img_embedding.tolist()],
            n_results=top_k
        )
        
        # Format the results
        hits = []
        for i, (id, doc, score) in enumerate(zip(
            results['ids'][0], 
            results['documents'][0], 
            results['distances'][0]
        )):
            # Only include results with reasonable similarity
            if score < 1.5:  # Adjust threshold as needed (lower is more similar)
                hits.append({
                    'content': doc,
                    'score': float(score),
                    'meta': results['metadatas'][0][i] if 'metadatas' in results else {}
                })
        return hits
    except Exception as e:
        print(f"Error in image retrieval: {str(e)}")
        return []