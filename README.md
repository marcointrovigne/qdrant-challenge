# Qdrant Hybrid Search Implementation

This project implements a hybrid search system using Qdrant vector database, combining sparse vectors (BM25), dense vectors (Instructor-XL), and late interaction reranking (ColBERT) for optimal search results.

## Technical Overview

The implementation features:
- Hybrid search combining multiple embedding approaches:
  - BM25 for sparse embeddings
  - Instructor-XL for dense embeddings
  - ColBERT for late interaction reranking
- Binary quantization for efficient dense vector storage
- User-based filtering capabilities
- Distributed setup with 2 nodes, 3 shards, and replication factor of 2
- Processing of 1M arXiv paper titles

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Hugging Face account and access token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marcointrovigne/qdrant-challenge.git
cd qdrant-challenge
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Log in to Hugging Face Hub:
```bash
huggingface-cli login
```
When prompted, enter your Hugging Face access token. You can create one at https://huggingface.co/settings/tokens

## Setting Up Qdrant

1. Start the Qdrant cluster using Docker Compose:
```bash
docker-compose up -d
```

This will create a two-node Qdrant cluster with:
- Primary node accessible at http://localhost:6333
- Secondary node for replication
- 3 shards distributed across nodes
- Replication factor of 2 for high availability

## Usage

The script supports several operations:

1. Create collection:
```bash
python main.py --action create
```

2. Upload data:
```bash
python main.py --action upload
```

3. Perform search:
```bash
python main.py --action search --query "your search query" [--user-id user_001]
```

4. Create collection and upload data in one command:
```bash
python main.py --action create_and_upload --force-delete
```

### Additional Options

- `--force-delete`: Force delete existing collection before creation
- `--user-id`: Filter search results by user ID (optional)

## Project Structure

- `main.py`: Entry point and command-line interface
- `qdrant_logic.py`: Core implementation of Qdrant operations
- `docker-compose.yml`: Qdrant cluster configuration
- `requirements.txt`: Python dependencies

## System Requirements

- Memory: Minimum 16GB RAM recommended due to embedding models
- Storage: At least 10GB free space for dataset and embeddings
- CPU: Multi-core processor recommended for embedding generation

## Dependencies

Key dependencies include:
- qdrant-client
- InstructorEmbedding
- fastembed
- datasets
- loguru

## Notes

- The system uses the arXiv titles dataset with pre-computed embeddings with Instructor-XL
- First-time setup may take longer due to model downloads
- The system supports 10 different user IDs (user_000 to user_009)

## Troubleshooting

1. If Qdrant connection fails:
   - Ensure Docker containers are running: `docker ps`
   - Check Qdrant logs: `docker-compose logs`

2. If dataset download fails:
   - Verify Hugging Face authentication

3. If memory issues occur:
   - Reduce batch_size in QdrantConfig
   - Ensure sufficient system memory