import json, os
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')

def chunk_text(meta_path, chunk_size=300):
    with open(meta_path) as f:
        meta = json.load(f)
    
    chunks = []
    buf = []
    cnt = 0
    
    for item in meta:
        if 'content' in item:
            sentences = sent_tokenize(item['content'])
            for sent in sentences:
                if len(buf) > 0 and cnt + len(sent) > chunk_size:
                    # Create chunk with initial ID
                    chunk = {
                        'chunk_id': f"{item.get('source', 'unknown')}_{item.get('page', '0')}",
                        'content': " ".join(buf),
                        'source': item.get('source', ''),
                        'page': item.get('page', 0),
                        'type': item.get('type', 'text')
                    }
                    chunks.append(chunk)
                    buf = []
                    cnt = 0
                buf.append(sent)
                cnt += len(sent)
    
    # Don't forget the last buffer
    if buf:
        chunk = {
            'chunk_id': f"{item.get('source', 'unknown')}_{item.get('page', '0')}",
            'content': " ".join(buf),
            'source': item.get('source', ''),
            'page': item.get('page', 0),
            'type': item.get('type', 'text')
        }
        chunks.append(chunk)
    
    # Add unique suffix to each chunk_id to avoid duplicates
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = f"{chunk['chunk_id']}_chunk{i}"
    
    out_path = os.path.join(os.path.dirname(meta_path), 'chunks.json')
    with open(out_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    return chunks