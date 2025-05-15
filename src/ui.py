import os
import streamlit as st
import tempfile
import json
import torch
from pathlib import Path
from data_extraction import extract_text_and_images
from preprocessing import chunk_text
from embedding import embed_texts, embed_images
from vectorstore import add_images
from vectorstore import add_texts, collection
from retrieval import retrieve_text, retrieve_image
from llm_integration import generate_answer
import time
from evaluation import evaluate_retrieval_quality, evaluate_text_quality, plot_embeddings, visualize_embeddings
import matplotlib.pyplot as plt
import pandas as pd

# Set environment variable to avoid PyTorch/Streamlit file watcher issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Define path for tracking processed documents
PROCESSED_DOCS_FILE = 'data/processed_docs.json'
os.makedirs('data', exist_ok=True)

# Configure page
st.set_page_config(
    page_title="Multimodal RAG System", 
    page_icon="📚", 
    layout="wide"
)

st.title('📚 Multimodal RAG System')
st.markdown("### A system for retrieving and answering questions from FAST-NUCES documents")

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses:
    - PDF document processing with OCR
    - CLIP for multimodal embeddings
    - ChromaDB for vector storage
    - Smart chunk merging for better context
    """)

# Initialize tab
tab_main, tab_index, tab_query, tab_evaluate = st.tabs(["About", "Index Documents", "Ask Questions", "Evaluate & Visualize"])

with tab_index:
    st.header("Document Indexing")
    if st.button('Initialize & Index Documents'):
        # Check if ChromaDB already has documents
        existing_count = collection.count()
        
        # Load list of previously processed documents
        processed_docs = []
        if os.path.exists(PROCESSED_DOCS_FILE):
            with open(PROCESSED_DOCS_FILE, 'r') as f:
                processed_docs = json.load(f)
        
        # 1. Discover all PDFs under docs/
        pdf_paths = [str(p) for p in Path('docs').rglob('*.pdf')]
        
        if not pdf_paths:
            st.error("No PDF files found in the 'docs/' directory. Please add PDFs and try again.")
        else:
            # Find new documents that haven't been processed
            new_pdfs = [pdf for pdf in pdf_paths if pdf not in processed_docs]
            
            if not new_pdfs and existing_count > 0:
                st.success(f"All {len(pdf_paths)} documents already indexed. Using existing database with {existing_count} chunks.")
            else:
                if new_pdfs:
                    with st.status("Processing documents...", expanded=True) as status:
                        st.info(f"Found {len(new_pdfs)} new documents to process out of {len(pdf_paths)} total PDFs.")
                        all_meta = []
                        # 2. Extract text and OCR from new PDFs only

                        for i, pdf in enumerate(new_pdfs):
                            st.write(f"Processing {i+1}/{len(new_pdfs)}: {pdf}...")
                            try:
                                meta, image_paths = extract_text_and_images(pdf, 'data')
                                all_meta += meta
                                processed_docs.append(pdf)
                            except Exception as e:
                                st.error(f"Error processing {pdf}: {str(e)}")
                                continue  # Skip to next PDF if there's an error

                            # This should be outside the try block but after the except
                            if image_paths:
                                st.write(f"Processing {len(image_paths)} images from {pdf}...")
                                try:
                                    # Generate embeddings for images
                                    image_embs = embed_images(image_paths)
                                    # Create IDs and metadata for images
                                    image_ids = [f"img_{pdf}_{i}" for i in range(len(image_paths))]
                                    image_metas = [{
                                        'source': pdf, 
                                        'type': 'image', 
                                        'path': path
                                    } for path in image_paths]
                                    # Add to vector store
                                    add_images(image_ids, image_embs, image_metas)
                                except Exception as e:
                                    st.error(f"Error processing images from {pdf}: {str(e)}")
                        
                        # Save updated list of processed documents
                        with open(PROCESSED_DOCS_FILE, 'w') as f:
                            json.dump(processed_docs, f, indent=2)
                        
                        # 3. Save metadata then chunk text
                        if all_meta:
                            with open('data/metadata.json','w') as f:
                                json.dump(all_meta, f, indent=2)
                            
                            st.write("Chunking text...")
                            chunks = chunk_text('data/metadata.json')
                            
                            if not chunks:
                                st.warning("No text content was extracted from the PDFs.")
                            else:
                                # 4. Embed and store chunks
                                st.write("Creating embeddings...")
                                embs = embed_texts([c['content'] for c in chunks])
                                st.write("Adding to vector database...")
                                add_texts(chunks, embs)
                                status.update(label="Processing complete!", state="complete")
                                st.success(f'Indexed {len(chunks)} chunks from {len(new_pdfs)} new PDFs')
                        else:
                            status.update(label="No content extracted", state="error")
                            st.warning("No metadata was extracted from the documents.")
                else:
                    # This handles edge case where database is empty but no new documents
                    st.warning("No documents in database but also no new PDFs to process.")

with tab_query:
    st.header("Ask Questions")
    mode = st.radio('Query Mode', ['Text', 'Image'], horizontal=True)

    if mode == 'Text':
        q = st.text_input('Enter your question about FAST-NUCES:')
        search_timeout = st.slider("Search timeout (seconds)", 10, 60, 30)
        if st.button('Search', key="text_search") and q:
            col1, col2 = st.columns([2, 1])
            
            with st.spinner('🔎 Searching for information...'):
                search_progress = st.progress(0)
                for i in range(10):
                    time.sleep(0.1)
                    search_progress.progress((i+1)/10)
                
                try:
                    hits = retrieve_text(q, timeout=search_timeout)
                    search_progress.progress(1.0)
                    
                    # Check if hits is empty or None
                    if not hits:
                        st.warning("No relevant information found for your query. Please try a different question.")
                        st.stop()
                    
                    with col2:
                        # Show retrieved passages in expandable section
                        with st.expander("Source Documents", expanded=False):
                            st.markdown("### Retrieved Passages")
                            for i, h in enumerate(hits):
                                # Use get() to avoid KeyError if keys don't exist
                                source = h.get('meta', {}).get('source', 'Unknown source')
                                page = h.get('meta', {}).get('page', 'N/A')
                                content = h.get('content', 'No content available')
                                
                                st.markdown(f"**Source {i+1}:** {source} (Page {page})")
                                st.markdown(f"```\n{content[:300]}{'...' if len(content) > 300 else ''}\n```")
                                st.markdown("---")
                    
                    # Generate answer with error handling
                    with col1:
                        with st.spinner('💭 Generating answer...'):
                            # Create a progress bar for answer generation
                            answer_progress = st.progress(0)
                            # Update progress periodically to show ongoing work
                            for i in range(10):
                                time.sleep(0.2)
                                answer_progress.progress((i+1)/10)
                            
                            try:
                                ans = generate_answer(hits, q)
                                answer_progress.progress(1.0)
                                st.markdown("### Answer")
                                st.markdown(ans)
                            except RuntimeError as e:
                                answer_progress.empty()
                                if "expected device" in str(e) or "CUDA out of memory" in str(e):
                                    st.error("Error: GPU memory issue detected. Try with a smaller query or restart the application.")
                                    # Fall back to showing just the retrieved passages
                                    st.markdown("### Unable to generate complete answer. Here are the key passages:")
                                    for i, h in enumerate(hits[:3]):
                                        st.markdown(f"**From {h.get('meta', {}).get('source', 'Unknown source')}**")
                                        st.markdown(h.get('content', 'No content available'))
                                else:
                                    st.error(f"Error generating answer: {str(e)}")
                            except Exception as e:
                                answer_progress.empty()
                                st.error(f"Unexpected error: {str(e)}")
                except Exception as e:
                    search_progress.empty()
                    st.error(f"Error retrieving information: {str(e)}")
                    
    else:
        st.write("Upload an image to find related information:")
        img = st.file_uploader('Upload image', type=['png','jpg','jpeg'])
        if st.button('Search', key="image_search") and img:
            col1, col2 = st.columns([2, 1])
            
            with st.spinner('Processing image...'):
                try:
                    # Save uploaded image to temporary file
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    tmp.write(img.read())
                    tmp.close()
                    
                    # Display the uploaded image
                    with col2:
                        st.image(tmp.name, caption="Uploaded image", use_container_width=True)
                    
                    # Retrieve similar content
                    hits = retrieve_image(tmp.name)
                    
                    # Check if hits is empty
                    if not hits:
                        st.warning("No relevant content found for this image.")
                        try:
                            os.unlink(tmp.name)  # Clean up temp file
                        except:
                            pass
                        st.stop()  # Stop execution to avoid further errors

                    with col2:
                        # Show retrieved passages in expandable section
                        with st.expander("Source Documents", expanded=False):
                            st.markdown("### Retrieved Passages")
                            for i, h in enumerate(hits):
                                # Use get() to avoid KeyError if keys don't exist
                                source = h.get('meta', {}).get('source', 'Unknown source')
                                page = h.get('meta', {}).get('page', 'N/A')
                                content = h.get('content', 'No content available')
                                
                                st.markdown(f"**Source {i+1}:** {source} (Page {page})")
                                st.markdown(f"```\n{content[:300]}{'...' if len(content) > 300 else ''}\n```")
                                st.markdown("---")
                    
                    # Generate answer with error handling
                    with col1:
                        with st.spinner('Generating analysis...'):
                            try:
                                ans = generate_answer(hits, 'Describe what this image might be related to based on the retrieved content.')
                                st.markdown("### Image Analysis")
                                st.markdown(ans)
                            except RuntimeError as e:
                                if "expected device" in str(e) or "CUDA out of memory" in str(e):
                                    st.error("Error: GPU memory issue detected. Try again or restart the application.")
                                    # Fall back to showing just the retrieved passages
                                    st.markdown("### Unable to generate complete analysis. Here are the key passages:")
                                    for i, h in enumerate(hits[:3]):
                                        st.markdown(f"**From {h.get('meta', {}).get('source', 'Unknown source')}**")
                                        st.markdown(h.get('content', 'No content available'))
                                else:
                                    st.error(f"Error generating analysis: {str(e)}")
                            except Exception as e:
                                st.error(f"Unexpected error: {str(e)}")
                    
                    # Cleanup
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

with tab_evaluate:
    st.header("System Evaluation & Visualization")
    
    eval_option = st.radio(
        "What would you like to evaluate?",
        ["Retrieval Quality", "Answer Quality", "Embedding Visualization"]
    )
    
    if eval_option == "Retrieval Quality":
        st.subheader("Retrieval Evaluation")
        st.write("This will evaluate the system's retrieval performance using Precision@K, Recall@K, and MAP metrics.")
        
        if st.button("Run Retrieval Evaluation"):
            with st.spinner("Evaluating retrieval quality..."):
                eval_results = evaluate_retrieval_quality()
                
                # Display results
                st.subheader("Evaluation Results")
                
                # Precision@K
                st.write("### Precision@K")
                precision_df = pd.DataFrame({
                    'K': list(eval_results['precision'].keys()),
                    'Precision': list(eval_results['precision'].values())
                })
                st.bar_chart(precision_df.set_index('K'))
                
                # Recall@K
                st.write("### Recall@K")
                recall_df = pd.DataFrame({
                    'K': list(eval_results['recall'].keys()),
                    'Recall': list(eval_results['recall'].values())
                })
                st.bar_chart(recall_df.set_index('K'))
                
                # MAP
                st.write(f"### Mean Average Precision (MAP): {eval_results['map']:.4f}")
    
    elif eval_option == "Answer Quality":
        st.subheader("Answer Quality Evaluation")
        st.write("This will evaluate the quality of generated answers against reference answers using BLEU and ROUGE metrics.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            query = st.text_area("Enter a query:", "What are the requirements for FYP submission?", height=100)
        
        with col2:
            reference = st.text_area("Enter reference answer:", "The FYP submission requires a final report, source code, and a presentation. The report must follow the provided template and be submitted one week before the demonstration.", height=100)
        
        if st.button("Evaluate Answer Quality"):
            with st.spinner("Generating and evaluating answer..."):
                # Get retrieved documents
                from retrieval import retrieve_text
                hits = retrieve_text(query, top_k=5)
                
                # Generate answer
                from llm_integration import generate_answer
                generated = generate_answer(hits, query)
                
                # Calculate metrics
                metrics = evaluate_text_quality(reference, generated)
                
                # Display results
                st.subheader("Generated Answer")
                st.write(generated)
                
                st.subheader("Evaluation Metrics")
                st.write(f"BLEU Score: {metrics['bleu']:.4f}")
                st.write(f"ROUGE-1 Score: {metrics['rouge-1']:.4f}")
                st.write(f"ROUGE-2 Score: {metrics['rouge-2']:.4f}")
                st.write(f"ROUGE-L Score: {metrics['rouge-l']:.4f}")
                
                # Display cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                from embedding import embed_texts
                
                ref_emb = embed_texts([reference])[0]
                gen_emb = embed_texts([generated])[0]
                
                similarity = cosine_similarity([ref_emb], [gen_emb])[0][0]
                st.write(f"Cosine Similarity: {similarity:.4f}")
    
    else:  # Embedding Visualization
        st.subheader("Embedding Space Visualization")
        st.write("This will generate a visualization of the document embedding space.")
        
        limit = st.slider("Number of documents to visualize", 100, 1000, 500)
        
        if st.button("Generate Visualization"):
            with st.spinner("Generating embedding visualization..."):
                viz_df = visualize_embeddings(limit=limit)
                
                if viz_df is not None:
                    # Plot and display
                    fig_path = plot_embeddings(viz_df)
                    
                    st.image(fig_path, caption="Embedding Space Visualization", use_container_width=True)
                    
                    st.subheader("Embedding Distribution")
                    st.write("Document types distribution:")
                    st.bar_chart(viz_df['document_type'].value_counts())
                    
                    st.write("Source documents distribution:")
                    st.bar_chart(viz_df['source'].value_counts())
                else:
                    st.error("No embeddings available for visualization. Please index some documents first.")