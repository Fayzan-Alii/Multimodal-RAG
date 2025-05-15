import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch
import json
import os
from collections import defaultdict
import pandas as pd
import umap
from embedding import embed_texts, text_model
from vectorstore import collection

# Initialize metrics calculator
rouge = Rouge()

def calculate_precision_at_k(relevant_docs, retrieved_docs, k=5):
    """Calculate precision@k given relevant and retrieved documents."""
    retrieved_k = retrieved_docs[:k]
    if not retrieved_k:
        return 0.0
    return len(set(relevant_docs) & set(retrieved_k)) / len(retrieved_k)

def calculate_recall_at_k(relevant_docs, retrieved_docs, k=5):
    """Calculate recall@k given relevant and retrieved documents."""
    retrieved_k = retrieved_docs[:k]
    if not relevant_docs:
        return 0.0
    return len(set(relevant_docs) & set(retrieved_k)) / len(relevant_docs)

def calculate_map(relevance_judgments, retrieved_results):
    """Calculate Mean Average Precision for a set of queries."""
    aps = []
    for query_id, relevant_docs in relevance_judgments.items():
        if query_id not in retrieved_results:
            continue
        
        retrieved = retrieved_results[query_id]
        ap = 0.0
        num_relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_docs:
                num_relevant_retrieved += 1
                precision_at_i = num_relevant_retrieved / (i + 1)
                ap += precision_at_i
        
        if relevant_docs:
            ap /= len(relevant_docs)
        else:
            ap = 0
        
        aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0

def evaluate_retrieval_quality(test_queries=None, k_values=[1, 3, 5, 10]):
    """Evaluate retrieval quality using test queries and relevant documents."""
    print("Starting retrieval quality evaluation...")
    
    # Load or create test set
    test_data_path = 'data/retrieval_test_set.json'
    
    if test_queries is None:
        if not os.path.exists(test_data_path):
            print("Creating new test set...")
            create_test_set()
        
        try:
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
                print(f"Loaded test set with {len(test_data)} queries")
        except Exception as e:
            print(f"Error loading test data: {e}")
            print("Creating new test set...")
            test_data = create_test_set()
    else:
        test_data = test_queries
    
    # Initialize results
    results = {
        'precision': defaultdict(list),
        'recall': defaultdict(list),
    }
    
    # Make sure k_values are sorted
    k_values = sorted(k_values)
    max_k = max(k_values)
    
    # Evaluate each query
    for i, query_data in enumerate(test_data):
        query = query_data['query']
        relevant_docs = query_data['relevant_docs']
        
        print(f"Evaluating query {i+1}/{len(test_data)}: {query[:30]}...")
        print(f"  - Has {len(relevant_docs)} relevant documents")
        
        # Get retrieved documents from the system
        try:
            from retrieval import retrieve_text
            hits = retrieve_text(query, top_k=max_k)
            
            # Extract IDs from hits, handling different possible structures
            retrieved_docs = []
            for hit in hits:
                if isinstance(hit, dict):
                    # Try different possible locations of the ID
                    if 'id' in hit:
                        retrieved_docs.append(hit['id'])
                    elif 'chunk_id' in hit:
                        retrieved_docs.append(hit['chunk_id'])
                    elif 'meta' in hit and isinstance(hit['meta'], dict) and 'chunk_id' in hit['meta']:
                        retrieved_docs.append(hit['meta']['chunk_id'])
                    # If no ID found, try to use the whole hit as ID
                    else:
                        retrieved_docs.append(str(hit))
                else:
                    # If hit is not a dict, use it directly (unlikely but just in case)
                    retrieved_docs.append(str(hit))
            
            print(f"  - Retrieved {len(retrieved_docs)} documents")
            if not retrieved_docs:
                print("  - WARNING: No documents retrieved!")
            
            # Calculate metrics for each k
            for k in k_values:
                if k <= len(retrieved_docs):
                    precision_k = calculate_precision_at_k(relevant_docs, retrieved_docs, k)
                    recall_k = calculate_recall_at_k(relevant_docs, retrieved_docs, k)
                    
                    results['precision'][k].append(precision_k)
                    results['recall'][k].append(recall_k)
                    
                    print(f"  - P@{k}: {precision_k:.4f}, R@{k}: {recall_k:.4f}")
                else:
                    # If k is larger than number of retrieved docs
                    results['precision'][k].append(0.0)
                    results['recall'][k].append(0.0)
        except Exception as e:
            print(f"  - ERROR evaluating query: {str(e)}")
            # Add zeros for this query
            for k in k_values:
                results['precision'][k].append(0.0)
                results['recall'][k].append(0.0)
    
    # Calculate average metrics
    avg_results = {
        'precision': {k: np.mean(values) for k, values in results['precision'].items()},
        'recall': {k: np.mean(values) for k, values in results['recall'].items()}
    }
    
    # Calculate MAP
    try:
        print("Calculating MAP score...")
        query_relevance = {i: query_data['relevant_docs'] for i, query_data in enumerate(test_data)}
        
        # Get retrieved results for each query
        query_retrieved_ids = {}
        for q_id, query_data in enumerate(test_data):
            try:
                from retrieval import retrieve_text
                hits = retrieve_text(query_data['query'], top_k=max_k)
                retrieved_docs = []
                for hit in hits:
                    if isinstance(hit, dict):
                        if 'id' in hit:
                            retrieved_docs.append(hit['id'])
                        elif 'chunk_id' in hit:
                            retrieved_docs.append(hit['chunk_id'])
                        elif 'meta' in hit and isinstance(hit['meta'], dict) and 'chunk_id' in hit['meta']:
                            retrieved_docs.append(hit['meta']['chunk_id'])
                        else:
                            retrieved_docs.append(str(hit))
                    else:
                        retrieved_docs.append(str(hit))
                query_retrieved_ids[q_id] = retrieved_docs
            except Exception as e:
                print(f"Error getting retrieved docs for MAP: {e}")
                query_retrieved_ids[q_id] = []
        
        map_score = calculate_map(query_relevance, query_retrieved_ids)
        avg_results['map'] = map_score
        print(f"MAP score: {map_score:.4f}")
    except Exception as e:
        print(f"Error calculating MAP: {e}")
        avg_results['map'] = 0.0
    
    print("Evaluation complete")
    return avg_results

def create_test_set(num_queries=10):
    """Create a test set for retrieval evaluation with real relevant documents."""
    test_data = []
    
    # Get sample documents from collection to check available IDs
    try:
        results = collection.peek(limit=300)
    except Exception as e:
        print(f"Error fetching documents: {e}")
        results = {'ids': [], 'documents': [], 'metadatas': []}
    
    # Print message for debugging
    print(f"Found {len(results['ids'])} documents in collection")
    
    # Map document IDs to their content and metadata for easier lookup
    doc_map = {}
    for i, doc_id in enumerate(results['ids']):
        if i < len(results['documents']) and i < len(results['metadatas']):
            doc_map[doc_id] = {
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
    
    # Define queries with keyword-based approach to find relevant docs
    test_queries = [
        {"query": "What are the requirements for FYP submission?", "keywords": ["FYP", "submission", "requirements", "report", "deliverable"]},
        {"query": "What is the financial performance of the university?", "keywords": ["financial", "revenue", "budget", "income", "expense"]},
        {"query": "What are the key programs offered at FAST-NUCES?", "keywords": ["program", "degree", "offering", "course", "curriculum"]},
        {"query": "What facilities are available at the campus?", "keywords": ["facility", "campus", "library", "lab", "infrastructure"]},
        {"query": "What is the university's vision?", "keywords": ["vision", "mission", "goal", "objective", "philosophy"]},
        {"query": "What degrees does FAST offer?", "keywords": ["degree", "program", "BS", "MS", "PhD"]},
        {"query": "What are the research areas at FAST?", "keywords": ["research", "area", "focus", "project", "publication"]},
        {"query": "What is the admission process?", "keywords": ["admission", "apply", "application", "enrollment", "entry"]},
        {"query": "What are the key achievements from the last year?", "keywords": ["achievement", "accomplishment", "success", "award", "milestone"]},
        {"query": "How many faculty members are there in the Computer Science department?", "keywords": ["faculty", "staff", "professor", "teacher", "computer science"]}
    ][:num_queries]
    
    # For each query, find relevant documents based on keyword matching
    for query_info in test_queries:
        query = query_info["query"]
        keywords = query_info["keywords"]
        relevant_docs = []
        
        # Find documents containing the keywords
        for doc_id, doc_data in doc_map.items():
            content = doc_data['content'].lower()
            if any(keyword.lower() in content for keyword in keywords):
                relevant_docs.append(doc_id)
                # Limit to at most 5 relevant docs per query
                if len(relevant_docs) >= 5:
                    break
        
        # If no relevant docs found, use at least one placeholder
        if not relevant_docs and len(results['ids']) > 0:
            relevant_docs = [results['ids'][0]]
            print(f"No relevant docs found for: {query}, using placeholder")
        
        # Add to test data
        test_data.append({
            'query': query,
            'relevant_docs': relevant_docs
        })
        print(f"Added query: {query} with {len(relevant_docs)} relevant docs")
    
    # Save test set
    os.makedirs('data', exist_ok=True)
    with open('data/retrieval_test_set.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test set with {len(test_data)} queries")
    return test_data

def evaluate_text_quality(reference, generated):
    """Evaluate text generation quality using BLEU and ROUGE scores."""
    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)
    
    # Calculate ROUGE scores
    try:
        rouge_scores = rouge.get_scores(generated, reference)[0]
    except:
        rouge_scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    
    return {
        'bleu': bleu_score,
        'rouge-1': rouge_scores['rouge-1']['f'],
        'rouge-2': rouge_scores['rouge-2']['f'],
        'rouge-l': rouge_scores['rouge-l']['f']
    }

def visualize_embeddings(limit=1000):
    """Visualize embedding space using UMAP dimensionality reduction."""
    try:
        # Get documents and embeddings from the collection
        results = collection.peek(limit=limit)
        
        if 'embeddings' not in results or len(results['embeddings']) == 0:
            print("No embeddings found in the collection.")
            return None
        
        # Convert embeddings to numpy array and ensure they're not PyTorch tensors
        embeddings = np.array(results['embeddings'])
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Apply UMAP with error handling
        try:
            reducer = umap.UMAP(
                n_neighbors=15, 
                min_dist=0.1, 
                n_components=2, 
                random_state=42,
                metric='euclidean'
            )
            reduced_embeddings = reducer.fit_transform(embeddings)
        except Exception as e:
            print(f"UMAP reduction failed: {e}")
            # Fall back to PCA if UMAP fails
            from sklearn.decomposition import PCA
            print("Falling back to PCA...")
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create a dataframe for visualization
        viz_df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'document_type': [meta.get('type', 'unknown') for meta in results['metadatas']],
            'source': [meta.get('source', 'unknown').split('/')[-1] for meta in results['metadatas']]
        })
        
        return viz_df
    except Exception as e:
        print(f"Error in visualize_embeddings: {e}")
        return None

def plot_embeddings(viz_df=None):
    """Generate embedding space visualization."""
    if viz_df is None:
        viz_df = visualize_embeddings()
        
    if viz_df is None:
        return None
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot points colored by document type
    sns.scatterplot(
        x='x', y='y',
        hue='document_type',
        palette=sns.color_palette("hls", len(viz_df['document_type'].unique())),
        data=viz_df,
        legend="full",
        alpha=0.5
    )
    
    # Add title and labels
    plt.title('Document Embedding Space Visualization', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Save the plot
    os.makedirs('data/visualizations', exist_ok=True)
    plt.savefig('data/visualizations/embedding_space.png', dpi=300, bbox_inches='tight')
    
    return 'data/visualizations/embedding_space.png'