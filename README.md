# VLM API

## About

REST API for computing cross-modal similarity between images and text using the [ColPaLI](https://huggingface.co/vidore/colpali-v1.2) vision-language model (read more [here](https://huggingface.co/blog/manu/colpali)). Provides token-level attention visualization and similarity scoring.

Interested in managed options? Get in touch with us at hi@datafog.ai. 

## Overview
**ColPali** is a **vision-language model** developed to improve document retrieval, specifically for documents with rich visual elements. Traditional retrieval models often struggle with complex layouts that combine text, images, and tables, as they primarily rely on text embeddings. ColPali, however, uses images of document pages instead of extracting text, enabling it to "see" visual information directly.

### How It Works
1. **Page Embeddings:** Each document page is represented as an image and divided into patches. These patches are fed into a vision transformer and later projected into a language model to generate embeddings.
2. **Multi-Vector Retrieval with Late Interaction:** Instead of matching a query to the document at a simple text level, ColPali allows for a richer comparison by matching each query term to specific document patches, then aggregating scores for better results.


#### Example
Uploading the image below and querying "Find PageRank", you can see the attention map and similarity score for the word chunk "Rank" in the document:
![example](example.png)

## Core Features

- **Cross-Modal Similarity Analysis**: Computes attention-based similarity scores between image regions and text tokens
- **Token-Level Granularity**: Breaks down similarity analysis per token, enabling fine-grained understanding
- **Attention Visualization**: Generates heatmaps showing which image regions correspond to specific text tokens
- **Quantitative Metrics**: Returns max/avg similarity scores and top-K attention hotspots per token

## Getting Started

*Note*: You will need a [Hugging Face API token](https://huggingface.co/settings/tokens) to access the ColPaLI model.  You can accept the conditions as a logged-in user from accepting the conditions on the [PaliGemma-3b](https://huggingface.co/google/paligemma-3b-mix-448) model card. 

## API Endpoints

The API provides endpoints for analyzing images and PDFs:

### PDF Analysis

**POST /query/pdf**

Process PDF documents with semantic search and highlighting.

Request:
```bash
curl -X POST \
  -F "file=@document.pdf" \
  -F "query=text query" \
  -F "top_k=3" \
  http://localhost:8000/query/pdf \
  --output result.zip
```

Response:

ZIP file containing highlighted PDF and similarity scores titled result.zip

## Setup

```bash
git clone https://github.com/DataFog/vlm-api
cd vlm-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 --limit-max-requests 1000 --loop uvloop --reload

```



### Pros
- **Better Visual Context:** Captures layout, images, and charts that typical text-only models miss.
- **Efficient Indexing:** Requires fewer pre-processing steps than standard OCR-based methods, which often have errors.
- **Improved Performance:** Outperforms traditional models on visually complex documents like infographics, charts, and tables.

### Cons
- **Computational Cost:** Using vision models can be resource-intensive, especially when handling large corpora.
- **Query Complexity:** For simpler, text-heavy documents, this approach might be overkill compared to standard retrieval methods.

ColPali thus shines in contexts where visual elements are key to understanding the document, making it a strong alternative to traditional retrieval-augmented generation approaches in such scenarios.

## Technical Details

- Model: ColPaLI v1.2 (vidore/colpali-v1.2)
- Precision: bfloat16 (requires macOS 14.0+ for Apple Silicon)
- Auto device selection (CPU/CUDA)
- CORS enabled
- Async request handling
- Automatic temp file cleanup