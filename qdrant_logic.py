import random
from dataclasses import dataclass
from typing import Optional

from InstructorEmbedding import INSTRUCTOR
from datasets import load_dataset
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import SparseVector, ScoredPoint


@dataclass
class QdrantConfig:
    """Configuration class for Qdrant vector database setup and operations.

    Attributes:
        collection_name (str): Name of the Qdrant collection.
        qdrant_url (str): URL for the Qdrant server connection.
        vector_size (int): Dimension of the dense vector embeddings.
        late_interaction_vector_size (int): Dimension for late interaction embeddings.
        shard_number (int): Number of shards for the collection.
        total_rows (int): Total number of rows to process from the dataset.
        replication_factor (int): Number of replicas for each shard.
        batch_size (int): Size of batches for processing and uploading.
        prefetch_limit (int): Maximum number of results to prefetch.
        late_interaction_limit (int): Maximum number of results for late interaction.
        user_ids (list[str]): List of user IDs for data assignment.
    """
    collection_name: str = "qdrant-challenge"
    qdrant_url: str = "http://localhost:6333"
    vector_size: int = 768
    late_interaction_vector_size: int = 128
    shard_number: int = 3
    total_rows: int = 1000000
    replication_factor: int = 2
    batch_size: int = 100
    prefetch_limit: int = 50
    late_interaction_limit: int = 10
    user_ids: list[str] = None

    def __post_init__(self):
        if self.user_ids is None:
            self.user_ids = [f"user_{i:03d}" for i in range(10)]


class QdrantDataset:
    """Class for managing dataset operations with Qdrant vector database."""

    def __init__(self, config: QdrantConfig = None):
        """
        Initialize QdrantDataset with configuration.

        Args:
            config (QdrantConfig, optional): Configuration object. If None, uses default config.
        """
        self.config = config or QdrantConfig()
        self._initialize_clients()
        self._initialize_models()
        self._load_dataset()

    def _load_dataset(self):
        """Load the arXiv titles dataset with pre-computed embeddings."""
        try:
            self.ds = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings")
        except Exception as e:
            raise RuntimeError(f"Failed to download the dataset: {str(e)}")

    def _initialize_clients(self):
        """Initialize connection to Qdrant server."""
        try:
            self.client = QdrantClient(url="http://localhost:6333")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

    def _initialize_models(self):
        """Initialize embedding models for different vector representations."""
        try:
            # BM25 for sparse embeddings
            self.bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
            # Instructor-XL for dense embeddings
            self.instructor = INSTRUCTOR('hkunlp/instructor-xl', device="cpu")
            # ColBERT for late interaction embeddings
            self.late_interaction = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def _embed_bm25(self,
                    text: str) -> SparseVector:
        """
        Generate BM25 sparse embeddings for input text.

        Args:
            text (str): Input text to embed.

        Returns:
            SparseVector: Sparse vector representation of the text.
        """
        em = list(self.bm25.embed(documents=text))
        sparse_vector = SparseVector(indices=em[0].indices.tolist(), values=em[0].values.tolist())
        return sparse_vector

    def _instructor_embed(self,
                          text: str) -> list[float]:
        """
        Generate Instructor-XL embeddings for input text.

        Args:
            text (str): Input text to embed.

        Returns:
            list[float]: Dense vector representation of the text.
        """
        instruction = "Represent the Research Paper title for retrieval; Input:"
        return self.instructor.encode([[instruction, text]]).tolist()[0]

    def _colbert_late_embed(self,
                            text: str) -> list[list[float]]:
        """
        Generate ColBERT late interaction embeddings for input text.

        Args:
            text (str): Input text to embed.

        Returns:
            list[list[float]]: Late interaction embeddings.
        """
        return next(self.late_interaction.query_embed(text)).tolist()

    def _process_dataset(self):
        """
        Process dataset in batches and prepare points for upload.

        Yields:
            list[PointStruct]: Batch of processed points ready for upload.
        """
        for batch_start in range(0, self.config.total_rows, self.config.batch_size):
            batch = self.ds["train"].select(
                range(batch_start, min(batch_start + self.config.batch_size, self.config.total_rows))
            )

            # Process each row in the batch
            batch_processed = [models.PointStruct(
                id=row["id"],
                payload={
                    "text": row["title"],
                    "user_id": random.choice(self.config.user_ids)
                },
                vector={
                    "instructor-xl": row["vector"],
                    "bm25": models.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    "colbertv2.0": self._colbert_late_embed(text=row["title"]),
                }
            ) for row in batch
                if (sparse_vec := self._embed_bm25(text=row["title"]))
            ]

            yield batch_processed

    def create_collection(self,
                          force_delete: bool = False) -> None:
        """
        Create a new Qdrant collection with specified configuration.

        Args:
            force_delete (bool): If True, deletes existing collection if it exists.

        Raises:
            Exception: If collection exists and force_delete is False.
        """
        logger.info(f"Creating Qdrant collection {self.config.collection_name}")

        try:
            if self.client.collection_exists(collection_name=self.config.collection_name):
                if force_delete:
                    logger.warning(f"Deleting existing collection: {self.config.collection_name}")
                    self.client.delete_collection(collection_name=self.config.collection_name)
                    logger.info(f"Collection {self.config.collection_name} deleted")
                else:
                    raise Exception(f"Collection {self.config.collection_name} already exists")

            # Configure vector spaces for hybrid search with multiple vector types
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    "instructor-xl": models.VectorParams(size=self.config.vector_size, distance=models.Distance.COSINE),
                    "colbertv2.0": models.VectorParams(
                        size=self.config.late_interaction_vector_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(),
                },
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                ),
                shard_number=self.config.shard_number,
                replication_factor=self.config.replication_factor,
            )

            logger.info(f"Collection created successfully: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Collection creation failed: {str(e)}")
            raise

    def upload_batch(self) -> None:
        """
        Upload processed dataset batches to Qdrant collection.

        Handles batch processing and uploading, tracking progress and failures.
        """
        total_uploaded = 0
        failed_batches = []

        for batch_idx, batch_points in enumerate(self._process_dataset()):
            try:
                if not batch_points:
                    logger.warning(f"Batch {batch_idx} was empty, skipping")
                    continue

                self.client.upload_points(
                    collection_name=self.config.collection_name,
                    points=batch_points
                )
                total_uploaded += len(batch_points)
                logger.info(f"Uploaded {total_uploaded} points of {self.config.total_rows}")

            except Exception as e:
                logger.error(f"Failed to upload batch {batch_idx}: {str(e)}")
                failed_batches.append(batch_idx)

        if failed_batches:
            logger.error(f"Failed to upload batches: {failed_batches}")

    def hybrid_search(self,
                      query: str,
                      user_id: Optional[str] = None) -> list[ScoredPoint]:
        """
        Perform hybrid search using multiple embedding approaches.

        Combines BM25 sparse search, Instructor-XL dense search, and ColBERT
        late interaction search for optimal results.

        Args:
            query (str): Search query text.
            user_id (Optional[str]): Filter results by user ID if provided.

        Returns:
            list[ScoredPoint]: List of search results with scores.

        Raises:
            RuntimeError: If search operation fails.
        """
        try:
            logger.info("Performing hybrid search")

            # Generate embeddings for query using all three approaches
            bm25_embed = self._embed_bm25(text=query)
            instructor_embed = self._instructor_embed(text=query)
            late_vectors = self._colbert_late_embed(text=query)

            # Build filter conditions if user_id is provided
            filter_conditions = []
            if user_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                )

            # Configure Reciprocal Rank Fusion (RRF) for combining search results
            rrf_prefetch = models.Prefetch(
                filter=models.Filter(must=filter_conditions) if filter_conditions else None,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(indices=bm25_embed.indices, values=bm25_embed.values),
                        using="bm25",
                        limit=self.config.prefetch_limit,
                    ),
                    models.Prefetch(
                        query=instructor_embed,
                        using="instructor-xl",
                        limit=self.config.prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=self.config.prefetch_limit
            )

            # Perform final search with late interaction
            results = self.client.query_points(
                collection_name=self.config.collection_name,
                prefetch=[rrf_prefetch],
                query=late_vectors,
                using="colbertv2.0",
                with_payload=True,
                limit=self.config.late_interaction_limit
            )

            return results.points

        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")
