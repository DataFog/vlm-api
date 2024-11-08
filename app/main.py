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
from contextlib import asynccontextmanager, contextmanager
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
from functools import lru_cache
from datetime import datetime, timedelta
import time
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
import asyncio
from typing import List, Dict
import tempfile
import os
import json
import zipfile
from PIL import Image
import torch
import io
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import logging
from tqdm import tqdm
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from functools import partial
import asyncio
from itertools import islice
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

async def get_pdf_images_stream(pdf_path):
    reader = PdfReader(pdf_path)
    for page_num in range(len(reader.pages)):
        # Process one page at a time
        images = convert_from_path(
            pdf_path,
            first_page=page_num+1,
            last_page=page_num+1,
            dpi=200
        )
        yield images[0], reader.pages[page_num].extract_text()

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

# Add context managers for cleanup
@contextmanager
def manage_temp_resources():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        torch.cuda.empty_cache()  # Clear CUDA cache

class ModelCache:
    def __init__(self, ttl_minutes=60):
        self.model = None
        self.processor = None
        self.device = None
        self.nlp = None
        self.last_access = None
        self.ttl = timedelta(minutes=ttl_minutes)
        self._lock = asyncio.Lock()

    async def get_or_load(self):
        async with self._lock:
            now = datetime.now()
            
            # Check if cache is expired
            if (self.last_access and 
                now - self.last_access > self.ttl):
                self.clear()

            # Load if needed
            if not self.model:
                await self.load_models()
            
            self.last_access = now
            return (self.model, self.processor, self.device)

    async def load_models(self):
        logging.info("Loading models into cache...")
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN environment variable not set")
        
        login(token=hf_token)
        
        model_name = "vidore/colpali-v1.2"
        self.device = get_torch_device("auto")
        
        # Set tokenizer parallelism before loading models
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        
        # Pre-format processor with default tokens
        self.processor.default_image_token = "<image>"
        self.processor.default_bos_token = "<bos>"
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def clear(self):
        logging.info("Clearing model cache")
        if self.model:
            self.model.cpu()
        self.model = None
        self.processor = None
        self.nlp = None
        torch.cuda.empty_cache()

# Initialize cache
model_cache = ModelCache()

# Replace get_model dependency with cached version
async def get_model():
    """Dependency that provides cached access to model components."""
    try:
        return await model_cache.get_or_load()
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=503, detail="Model loading failed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages model lifecycle during application startup and shutdown."""
    try:
        # Warm up cache on startup
        await model_cache.get_or_load()
        yield
    finally:
        # Cleanup on shutdown
        model_cache.clear()
app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # Format query with proper tokens
        formatted_query = f"<image><bos>{query}"
        
        # Preprocess inputs
        print("Preprocessing inputs...")
        batch_images = processor.process_images([image]).to(device)
        batch_queries = processor.process_queries([formatted_query]).to(device)
        
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

# Add this new route for status checking
class ProcessingStatus:
    def __init__(self):
        self.current_status = {}
        
    def update_status(self, job_id: str, status: dict):
        self.current_status[job_id] = status
        
    def get_status(self, job_id: str) -> dict:
        return self.current_status.get(job_id, {})

processing_status = ProcessingStatus()

def process_page_chunk(
    pdf_path: str,
    page_numbers: List[int],
    dpi: int = 100  # Reduced DPI
) -> List[Image.Image]:
    """Process a chunk of PDF pages in parallel."""
    return convert_from_path(
        pdf_path,
        first_page=min(page_numbers) + 1,
        last_page=max(page_numbers) + 1,
        dpi=dpi,
        thread_count=1  # Use 1 thread per worker as we're already parallel
    )

def batch_generator(iterable, batch_size):
    """Generate batches from an iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

async def process_pdf_chunks(
    pdf_path: str,
    query: str,
    job_id: str,
    model,
    processor,
    device,
    chunk_size: int = 2,  # Reduced chunk size
    dpi: int = 100  # Reduced DPI
) -> List[Dict]:
    """Process PDF in chunks with parallel processing."""
    results = []
    
    # Format query properly with tokens
    formatted_query = f"<image><bos>{query}"
    
    # Get total number of pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    # Create page number chunks
    page_chunks = list(batch_generator(range(total_pages), chunk_size))
    
    # Process page chunks in parallel
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
        for chunk_idx, page_numbers in enumerate(page_chunks):
            processing_status.update_status(job_id, {
                "status": "processing",
                "progress": f"{chunk_idx * chunk_size}/{total_pages}",
                "message": f"Processing pages {min(page_numbers)+1} to {max(page_numbers)+1}"
            })
            
            try:
                # Process images in parallel
                process_func = partial(process_page_chunk, pdf_path, page_numbers, dpi)
                images = await asyncio.get_event_loop().run_in_executor(executor, process_func)
                
                # Extract text for these pages
                chunk_texts = [reader.pages[i].extract_text() for i in page_numbers]
                
                # Optimize batch processing
                batch_size = 2  # Process 2 images at a time to manage memory
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    
                    # Process images and query
                    batch_images_processed = processor.process_images(batch_images).to(device)
                    batch_queries = processor.process_queries([formatted_query] * len(batch_images)).to(device)
                    
                    with torch.no_grad():
                        image_embeddings = model.forward(**batch_images_processed)
                        query_embeddings = model.forward(**batch_queries)
                        
                        image_mask = processor.get_image_mask(batch_images_processed)
                        
                        for j, (image, embedding) in enumerate(zip(batch_images, image_embeddings)):
                            page_idx = page_numbers[i + j]
                            n_patches = processor.get_n_patches(
                                image_size=image.size,
                                patch_size=model.patch_size
                            )
                            
                            similarity_maps = get_similarity_maps_from_embeddings(
                                image_embeddings=embedding.unsqueeze(0),
                                query_embeddings=query_embeddings[j].unsqueeze(0),
                                n_patches=n_patches,
                                image_mask=image_mask[j:j+1]
                            )[0]
                            
                            results.append({
                                "page_num": page_idx,
                                "similarity": float(similarity_maps.max().item()),
                                "text": chunk_texts[i + j]
                            })
                    
                    # Clear GPU memory after each batch
                    torch.cuda.empty_cache()
                    await asyncio.sleep(0)
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                continue
        
        executor.shutdown(wait=True)
    return results

@app.post("/query/pdf")
async def query_pdf(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(3),
    dpi: int = Form(300),  # Allow DPI customization
    deps: tuple = Depends(get_model),
):
    """Process PDF with optimized performance."""
    model, processor, device = deps
    job_id = str(uuid.uuid4())
    
    # Format query properly with tokens
    formatted_query = f"<image><bos>{query}"
    
    processing_status.update_status(job_id, {
        "status": "starting",
        "progress": "0%",
        "message": "Initializing PDF processing"
    })
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save PDF
            pdf_path = os.path.join(temp_dir, "input.pdf")
            with open(pdf_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process PDF in optimized chunks
            page_results = await process_pdf_chunks(
                pdf_path=pdf_path,
                query=query,
                job_id=job_id,
                model=model,
                processor=processor,
                device=device,
                chunk_size=2,  # Process 2 pages at a time
                dpi=dpi
            )
            
            if not page_results:
                raise HTTPException(status_code=500, detail="No results generated from PDF processing")
            
            # Get top matches
            top_pages = sorted(
                page_results,
                key=lambda x: x["similarity"],
                reverse=True
            )[:top_k]
            
            # Prepare results
            results_data = {
                "query": query,
                "total_pages": len(page_results),
                "top_matches": []
            }
            
            # Generate heatmaps only for top matches in parallel
            async def process_top_match(page):
                page_num = page["page_num"]
                
                # Generate heatmap for this page
                image = convert_from_path(
                    pdf_path,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=dpi
                )[0]
                
                # Generate and save heatmap
                batch_image = processor.process_images([image]).to(device)
                batch_query = processor.process_queries([formatted_query]).to(device)
                
                with torch.no_grad():
                    image_embedding = model.forward(**batch_image)
                    query_embedding = model.forward(**batch_query)
                    
                    n_patches = processor.get_n_patches(
                        image_size=image.size,
                        patch_size=model.patch_size
                    )
                    image_mask = processor.get_image_mask(batch_image)
                    
                    similarity_maps = get_similarity_maps_from_embeddings(
                        image_embeddings=image_embedding,
                        query_embeddings=query_embedding,
                        n_patches=n_patches,
                        image_mask=image_mask,
                    )[0]
                
                # Save heatmap
                heatmap_path = os.path.join(temp_dir, f"page_{page_num}.png")
                fig, ax = plot_similarity_map(
                    image=image,
                    similarity_map=similarity_maps.max(dim=0).values,
                    figsize=(8, 8),
                    show_colorbar=False,
                )
                fig.savefig(heatmap_path, bbox_inches="tight")
                plt.close(fig)
                
                return {
                    "page_number": page_num,
                    "similarity_score": float(page["similarity"]),
                    "page_text": page["text"]
                }
            
            # Process top matches in parallel
            tasks = [process_top_match(page) for page in top_pages]
            results_data["top_matches"] = await asyncio.gather(*tasks)
            
            # Create ZIP file
            zip_path = os.path.join(temp_dir, "results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add results.json
                results_json_path = os.path.join(temp_dir, "results.json")
                with open(results_json_path, 'w') as f:
                    json.dump(results_data, f, indent=2)
                zipf.write(results_json_path, "results.json")
                
                # Add heatmaps
                for filename in os.listdir(temp_dir):
                    if filename.startswith("page_") and filename.endswith(".png"):
                        zipf.write(
                            os.path.join(temp_dir, filename),
                            f"heatmaps/{filename}"
                        )
            
            # Stream the response
            return StreamingResponse(
                open(zip_path, "rb"),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=results.zip"}
            )
            
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    finally:
        if job_id in processing_status.current_status:
            del processing_status.current_status[job_id]

# Add status endpoint
@app.get("/query/pdf/status/{job_id}")
async def get_processing_status(job_id: str):
    status = processing_status.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    # Add cleanup handler
    multiprocessing.set_start_method('fork')  # Use 'spawn' on Windows
    
    # Modify server config
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        workers=2, 
        timeout=300, 
        limit_max_requests=1000,
        loop="uvloop"  # More efficient event loop
    )
