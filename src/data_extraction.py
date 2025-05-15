import fitz
from pdf2image import convert_from_path
import pytesseract, json, os

def extract_text_and_images(pdf_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    meta = []
    image_paths = []
    
    # Get filename without extension
    base_filename = os.path.basename(pdf_path).rsplit('.', 1)[0]
    
    for i, page in enumerate(doc):
        # Extract text
        text = page.get_text('text')
        if text.strip():
            meta.append({
                'source': pdf_path,
                'page': i,
                'type': 'text',
                'content': text
            })
            
        # Extract images
        images = page.get_images(full=True)
        for j, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save image to disk
            image_filename = f"{base_filename}_page{i+1}_img{j+1}.png"
            image_path = os.path.join(out_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # Store image metadata
            meta.append({
                'source': pdf_path,
                'page': i,
                'type': 'image',
                'content': f"Image on page {i+1}",
                'image_path': image_path
            })
            image_paths.append(image_path)
            
        # OCR images
        img = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)[0]
        ocr = pytesseract.image_to_string(img)
        if ocr.strip():
            meta.append({
                'source': pdf_path, 
                'page': i,
                'type': 'ocr',
                'content': ocr
            })
            
    with open(os.path.join(out_dir,'metadata.json'),'w') as f:
        json.dump(meta, f, indent=2)
    
    return meta, image_paths