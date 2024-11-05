"""FastAPI service for the VLM vision-language model that provides similarity analysis between images and text queries.

This module implements a REST API for analyzing similarity between images and text queries using the ColPaLI model.
It provides endpoints for single image queries and heatmap generation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
    global model, processor, device
    
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
    
    yield
    
    # Shutdown cleanup
    model = None
    processor = None

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

@app.post("/query/single/heatmap")
async def query_single_image_heatmap(
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
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)