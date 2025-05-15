import torch
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer('clip-ViT-B-32')
vision_model, preprocess = clip.load('ViT-B/32', device=device)

def embed_texts(texts):
    if not texts:  # Handle empty text list
        return []
    return text_model.encode(texts, convert_to_numpy=True)

def embed_images(paths):
    if not paths:  # Handle empty path list
        return []
    try:
        imgs = [preprocess(Image.open(p).convert('RGB')).unsqueeze(0).to(device) for p in paths]
        batch = torch.cat(imgs)
        with torch.no_grad(): 
            embs = vision_model.encode_image(batch)
        return embs.cpu().numpy()
    except Exception as e:
        print(f"Error embedding images: {str(e)}")
        return []

def embed_image(img_path):
    """Embed a single image."""
    try:
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = vision_model.encode_image(img)
        return emb.cpu().numpy()[0]  # Return the vector, not batch
    except Exception as e:
        print(f"Error embedding image {img_path}: {str(e)}")
        return None