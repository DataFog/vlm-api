# """FastAPI service for the VLM vision-language model that provides similarity analysis between images and text queries.

# This module implements a REST API for analyzing similarity between images and text queries using the ColPaLI model.
# It provides endpoints for single image queries and heatmap generation.
# """

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from PIL import Image
import io
from typing import List, Optional
from contextlib import asynccontextmanager
from huggingface_hub import login
import os
import logging
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
from colpali_engine.interpretability import plot_similarity_map, plot_all_similarity_maps
import tempfile
import os
import zipfile
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from colpali_engine.interpretability import get_similarity_maps_from_embeddings
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from PyPDF2 import PdfReader, PdfWriter
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks
from PIL import Image
import torch
import numpy as np
from typing import List, Tuple, Dict
import tempfile
import os
import zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from PyPDF2 import PdfReader, PdfWriter
import io
from pdf2image import convert_from_path
import spacy
from transformers import AutoTokenizer
import json
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Convert response content to BytesIO object for in-memory PDF handling
        # Returns: BytesIO object containing the downloaded PDF content
        # Requires: response.content is valid PDF data
        # Ensures: Returns a readable BytesIO object
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")


def get_pdf_images(pdf_path, is_local=True):
    """Get images and text from PDF file."""
    try:
        if not is_local:
            pdf_file = download_pdf(pdf_path)
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())
            pdf_path = "temp.pdf"
        
        # Extract text first
        reader = PdfReader(pdf_path)
        page_texts = []
        for page in reader.pages:
            text = page.extract_text()
            page_texts.append(text)
        
        # Convert to images with specific DPI and use poppler
        images = convert_from_path(
            pdf_path,
            dpi=200,  # Adjust DPI as needed
            poppler_path=None,  # Set this to your poppler path if needed
            fmt='PIL'
        )
        
        if not images:
            raise ValueError("No images extracted from PDF")
            
        print(f"Extracted {len(images)} images and {len(page_texts)} text pages")
        return images, page_texts
        
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global model variables
model = None  # ColPaLI model instance
processor = None  # ColPaLI processor instance  
device = None  # Torch device (CPU/GPU)

class QueryRequest(BaseModel):
    """Request model for query endpoints.
    
    Attributes:
        query (str): Text query to analyze against image
        token_idx (Optional[int]): Index of specific token to analyze
    """
    query: str
    token_idx: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages model lifecycle during application startup and shutdown.
    
    Requires:
        - HF_TOKEN environment variable must be set
        
    Effects:
        - Initializes global model, processor and device on startup
        - Cleans up resources on shutdown
        
    Args:
        app (FastAPI): FastAPI application instance
    """
    global model, processor, device, nlp
    
    # Startup
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")
    
    login(token=hf_token)
    
    model_name = "vidore/colpali-v1.2"
    device = get_torch_device("auto")
    
    print(f"Loading model on device: {device}")
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained(model_name)
    
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Installing spacy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    yield
    
    # Shutdown cleanup
    model = None
    processor = None
    nlp = None
app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_model():
    """Dependency that provides access to model components.
    
    Returns:
        tuple: (model, processor, device) tuple
        
    Raises:
        HTTPException: If model is not loaded
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model, processor, device

@app.post("/query/single")
async def query_image(
    file: UploadFile = File(...),
    query: str = Form(...),
    deps: tuple = Depends(get_model)
):
    """Analyze similarity between an image and text query.
    
    Requires:
        - Valid image file
        - Non-empty query string
        - Model must be loaded
        
    Effects:
        - Processes image and query through model
        - Generates similarity maps
        
    Args:
        file (UploadFile): Image file to analyze
        query (str): Text query to compare against image
        deps (tuple): Model dependencies from get_model()
        
    Returns:
        dict: Analysis results containing:
            - query: Original query text
            - n_patches: Image patch dimensions
            - tokens: Token-level similarity data
            
    Raises:
        HTTPException: On processing errors
    """
    model, processor, device = deps
    
    try:
        print(f"Processing image analysis request with query: {query}")
        
        # Read and validate image
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))
        print(f"Loaded image with size: {image.size}")
        
        # Preprocess inputs
        print("Preprocessing inputs...")
        batch_images = processor.process_images([image]).to(device)
        batch_queries = processor.process_queries([query]).to(device)
        
        # Generate embeddings
        print("Running model inference...")
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            query_embeddings = model.forward(**batch_queries)
        
        # Calculate similarity maps
        print("Generating similarity maps...")
        n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size)
        n_patches_list = [int(n) for n in n_patches]
        image_mask = processor.get_image_mask(batch_images)
        
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )
        
        similarity_maps = batched_similarity_maps[0]
        
        # Process query tokens
        print("Processing query tokens...")
        query_content = processor.decode(batch_queries.input_ids[0]).replace(processor.tokenizer.pad_token, "")
        query_content = query_content.replace(processor.query_augmentation_token, "").strip()
        query_tokens = processor.tokenizer.tokenize(query_content)
        
        # Build response
        token_similarities = {}
        for idx, token in enumerate(query_tokens):
            similarity_map = similarity_maps[idx]
            token_similarities[token] = {
                "token_idx": idx,
                "max_similarity_score": float(similarity_map.max().item()),
                "avg_similarity_score": float(similarity_map.mean().item()),
                "hotspots": [  # Only return top K most relevant areas
                    {
                        "x": int(x),
                        "y": int(y),
                        "score": float(similarity_map[x,y].item())
                    }
                    for x, y in torch.topk(similarity_map.view(-1), k=5)[1].tolist()
                ]
            }
        
        print("Analysis completed successfully")
        return {
            "query": query,
            "overall_similarity": float(similarity_maps.max().item()),
            "tokens": token_similarities
        }
        
    except Exception as e:
        print(f"Error during image analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity/heatmaps")
async def generate_similarity_heatmaps(
    file: UploadFile = File(...),
    query: str = Form(...),
    token_idx: Optional[int] = Form(None),
    deps: tuple = Depends(get_model)
):
    """Generate similarity heatmap visualization(s) for image-query pair.
    
    Requires:
        - Valid image file
        - Non-empty query string
        - Model must be loaded
        - If token_idx provided, must be valid index
        
    Effects:
        - Creates temporary files for heatmap images
        - Cleans up temporary files after response
        
    Args:
        file (UploadFile): Image file to analyze
        query (str): Text query to compare against image
        token_idx (Optional[int]): Specific token to visualize, if None generates all
        deps (tuple): Model dependencies from get_model()
        
    Returns:
        FileResponse: PNG image file for single heatmap or ZIP file for all tokens
        
    Raises:
        HTTPException: On processing errors
    """
    model, processor, device = deps
    
    # Create temporary directory for files
    temp_dir = tempfile.mkdtemp()
    try:
        # Process image and query
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))
        
        batch_images = processor.process_images([image]).to(device)
        batch_queries = processor.process_queries([query]).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            query_embeddings = model.forward(**batch_queries)
        
        # Calculate similarity maps
        n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size)
        image_mask = processor.get_image_mask(batch_images)
        
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )
        
        similarity_maps = batched_similarity_maps[0]
        
        # Process query tokens
        query_content = processor.decode(batch_queries.input_ids[0]).replace(processor.tokenizer.pad_token, "")
        query_content = query_content.replace(processor.query_augmentation_token, "").strip()
        query_tokens = processor.tokenizer.tokenize(query_content)

        if token_idx is not None:
            # Generate single token heatmap
            current_similarity_map = similarity_maps[token_idx]
            fig, ax = plot_similarity_map(
                image=image,
                similarity_map=current_similarity_map,
                figsize=(8, 8),
                show_colorbar=False,
            )
            max_sim_score = similarity_maps[token_idx, :, :].max().item()
            ax.set_title(f"Token #{token_idx}: `{query_tokens[token_idx]}`. MaxSim score: {max_sim_score:.2f}", fontsize=14)
            
            output_path = os.path.join(temp_dir, "heatmap.png")
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            
            # Setup cleanup tasks
            background_tasks = BackgroundTasks()
            background_tasks.add_task(lambda: os.remove(output_path))
            background_tasks.add_task(lambda: os.rmdir(temp_dir))
            
            return FileResponse(
                output_path, 
                media_type="image/png",
                background=background_tasks
            )
        else:
            # Generate heatmaps for all tokens
            plots = plot_all_similarity_maps(
                image=image,
                query_tokens=query_tokens,
                similarity_maps=similarity_maps,
                figsize=(8, 8),
                show_colorbar=False,
                add_title=True,
            )
            
            zip_path = os.path.join(temp_dir, "heatmaps.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for idx, (fig, ax) in enumerate(plots):
                    img_path = os.path.join(temp_dir, f"heatmap_{idx}.png")
                    fig.savefig(img_path, bbox_inches="tight")
                    plt.close(fig)
                    zipf.write(img_path, os.path.basename(img_path))
                    os.remove(img_path)
            
            # Setup cleanup tasks
            background_tasks = BackgroundTasks()
            background_tasks.add_task(lambda: os.remove(zip_path))
            background_tasks.add_task(lambda: os.rmdir(temp_dir))
            
            return FileResponse(
                zip_path, 
                media_type="application/zip",
                filename="heatmaps.zip",
                background=background_tasks
            )
                
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        print(f"Error generating heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


def extract_key_terms(query: str, nlp) -> List[str]:
    """Extract important terms from query using spaCy."""
    doc = nlp(query)
    key_terms = []
    
    # Extract named entities
    key_terms.extend([ent.text for ent in doc.ents])
    
    # Extract noun phrases and numbers
    key_terms.extend([chunk.text for chunk in doc.noun_chunks])
    key_terms.extend([token.text for token in doc if token.like_num])
    
    # Remove duplicates while preserving order
    seen = set()
    key_terms = [x for x in key_terms if not (x in seen or seen.add(x))]
    
    return key_terms

def create_highlighted_pdf(original_pdf_path: str, 
                         highlights: List[Dict],
                         output_path: str):
    """Create a new PDF with highlights overlaid on matching pages."""
    reader = PdfReader(original_pdf_path)
    writer = PdfWriter()
    
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        page_highlights = [h for h in highlights if h["page_num"] == page_num]
        
        if page_highlights:
            # Create highlight layer
            packet = io.BytesIO()
            can = canvas.Canvas(packet)
            
            for highlight in page_highlights:
                # Convert normalized coordinates to PDF coordinates
                x, y = highlight["bbox"]
                width = page.mediabox.width
                height = page.mediabox.height
                
                # Draw semi-transparent yellow highlight
                can.setFillColor(Color(1, 1, 0, alpha=0.3))
                can.rect(x * width, y * height, 
                        highlight["width"] * width,
                        highlight["height"] * height,
                        fill=True, stroke=False)
            
            can.save()
            packet.seek(0)
            
            # Merge highlight layer with original page
            highlight_pdf = PdfReader(packet)
            page.merge_page(highlight_pdf.pages[0])
        
        writer.add_page(page)
    
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

@app.post("/query/pdf")
async def query_pdf(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(3),
    deps: tuple = Depends(get_model)
):
    """
    Process PDF query with semantic search and highlighting.
    
    Args:
        file: PDF file to analyze
        query: Search query
        top_k: Number of top matches to return
        deps: Model dependencies
        
    Returns:
        ZIP file containing highlighted PDF and similarity scores
    """
    model, processor, device = deps
    
    try:
        # Load spaCy for query analysis
        nlp = spacy.load("en_core_web_sm")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, "input.pdf")
        
        # Save uploaded PDF
        pdf_content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
            
        # Convert PDF to images and extract text
        images, page_texts = get_pdf_images(pdf_path, is_local=True)
        
        # Extract key terms from query
        key_terms = extract_key_terms(query, nlp)
        print(f"Extracted key terms: {key_terms}")
        
        # Process each page
        page_results = []
        
        for page_num, (image, page_text) in enumerate(zip(images, page_texts)):
            try:
                # Create batch for image
                batch_images = processor.process_images([image]).to(device)
                batch_queries = processor.process_queries([query]).to(device)
                
                # Generate embeddings
                with torch.no_grad():
                    image_embeddings = model.forward(**batch_images)
                    query_embeddings = model.forward(**batch_queries)
                
                # Calculate similarity maps
                n_patches = processor.get_n_patches(image_size=image.size, 
                                                  patch_size=model.patch_size)
                image_mask = processor.get_image_mask(batch_images)
                
                similarity_maps = get_similarity_maps_from_embeddings(
                    image_embeddings=image_embeddings,
                    query_embeddings=query_embeddings,
                    n_patches=n_patches,
                    image_mask=image_mask,
                )[0]
                
                # Calculate overall similarity as max across all tokens
                overall_similarity = float(similarity_maps.max().item())
                
                # Generate heatmap visualization
                fig, ax = plot_similarity_map(
                    image=image,
                    similarity_map=similarity_maps.max(dim=0)[0],  # Combine all token maps
                    figsize=(12, 12),
                    show_colorbar=True
                )
                
                # Save heatmap image
                heatmap_path = os.path.join(temp_dir, f"heatmap_{page_num}.png")
                fig.savefig(heatmap_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                
                page_results.append({
                    "page_num": page_num,
                    "similarity": overall_similarity,
                    "text": page_text,
                    "heatmap_path": heatmap_path
                })
                
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue
        
        # Sort by similarity and get top_k pages
        top_pages = sorted(page_results, key=lambda x: x["similarity"], 
                         reverse=True)[:top_k]
        
        # Create results JSON
        results = {
            "top_matches": [
                {
                    "page_num": page["page_num"],
                    "similarity": page["similarity"],
                    "text_preview": page["text"][:200] + "..."
                }
                for page in top_pages
            ]
        }
        
        # Save results
        with open(os.path.join(temp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Create ZIP with results
        zip_path = os.path.join(temp_dir, "results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add results JSON
            zipf.write(os.path.join(temp_dir, "results.json"), "results.json")
            
            # Add heatmap images for top matches
            for page in top_pages:
                heatmap_name = f"heatmap_{page['page_num']}.png"
                zipf.write(page["heatmap_path"], heatmap_name)
        
        # Setup cleanup
        background_tasks = BackgroundTasks()
        background_tasks.add_task(lambda: shutil.rmtree(temp_dir))
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="results.zip",
            headers={"Content-Disposition": "attachment; filename=results.zip"},
            background=background_tasks
        )

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error in PDF processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
