import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from django.conf import settings
from django.utils import timezone

from .faiss_service import FAISSService
from ..models import DocumentChunk

logger = logging.getLogger(__name__)

# Global manager instance (singleton)
_vector_store_manager = None


def get_vector_store_manager():
    """Get the global VectorStoreManager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


class VectorStoreManager:
    """
    FAISS-based vector store manager.
    - Manages the cache of loaded indexes (FAISSService) for collections.
    - Provides high-level methods for indexing, search and deletion.
    - Keeps compatibility with existing code (tasks.py).
    """

    def __init__(self):
        self.config = getattr(settings, 'VECTOR_STORE_CONFIG', {
            'type': 'faiss',
            'dimension': 384,
            'metric': 'cosine',
            'use_gpu': False,
            'index_path': str(settings.FAISS_INDEX_PATH)
        })
        self._index_cache: Dict[int, FAISSService] = {}  # collection_id -> FAISSService
        self._default_dimension = self.config.get('dimension', 384)
        self._default_metric = self.config.get('metric', 'cosine')

    def _get_faiss_service(self, collection_id: Optional[int] = None) -> FAISSService:
        """
        Get a FAISSService for the given collection.
        If no collection is given, a new service without an index is created.
        """
        if collection_id is None:
            return FAISSService()

        if collection_id not in self._index_cache:
            svc = FAISSService()
            if svc.load_index(collection_id):
                self._index_cache[collection_id] = svc
            else:
                # Index not loaded, but return the empty service — it can be created later
                self._index_cache[collection_id] = svc
        return self._index_cache[collection_id]

    # ----------------------------------------------------------------------
    # Chunk indexing
    # ----------------------------------------------------------------------
    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_id: Optional[int] = None
    ) -> int:
        """
        Index a list of chunks into FAISS.
        - chunks: list of chunks (must have embedding_vector).
        - collection_id: collection ID (required to save the index).
        Returns the number of vectors successfully added.
        """
        if not chunks:
            return 0

        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.has_embedding and c.embedding_vector is not None]
        if not valid_chunks:
            logger.warning("No chunks with embeddings to index")
            return 0

        # Collect vectors and IDs
        vectors = []
        chunk_ids = []
        dimension = None

        for chunk in valid_chunks:
            vec = chunk.embedding_vector
            if isinstance(vec, list):
                vec = np.array(vec, dtype='float32')
            if dimension is None:
                dimension = len(vec)
            elif len(vec) != dimension:
                logger.error(f"Chunk {chunk.id} has inconsistent dimension {len(vec)} != {dimension}")
                continue
            vectors.append(vec)
            chunk_ids.append(chunk.id)

        if not vectors:
            return 0

        vectors_array = np.stack(vectors).astype('float32')

        # Get or create the index for the collection
        if collection_id is not None:
            svc = self._get_faiss_service(collection_id)
            # If the index does not exist, create it
            if svc.index is None:
                svc.create_index(
                    dimension=dimension,
                    index_type='flat',  # could be parameterized, flat for simplicity
                    metric=self._default_metric
                )
            # Add the vectors
            added = svc.add_vectors(vectors_array, chunk_ids)
            # Save the index to disk
            svc.save_index(collection_id)
            return added
        else:
            # Without a collection, do not save, just return the count
            logger.warning("Indexing without collection_id: not saved to disk")
            return len(vectors)

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        collection_id: Optional[int] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the vector index.
        - query_embedding: query vector (1D or 2D).
        - collection_id: collection ID (required).
        - k: number of results.
        - filters: ignored in this implementation (kept for compatibility).
        Returns a list of results with keys:
            - chunk_id, score, metadata (contains chunk_id, document_id, etc.)
        """
        if collection_id is None:
            logger.error("Search requires collection_id")
            return []

        svc = self._get_faiss_service(collection_id)
        if svc.index is None or svc.index.ntotal == 0:
            logger.warning(f"No index loaded for collection {collection_id}")
            return []

        # Run the search
        results = svc.search(query_embedding, k=k)

        # Format the result
        formatted = []
        for chunk_id, score in results:
            formatted.append({
                'chunk_id': chunk_id,
                'score': score,
                'metadata': {
                    'chunk_id': chunk_id,
                    'collection_id': collection_id,
                }
            })
        return formatted

    # ----------------------------------------------------------------------
    # Chunk removal from index
    # ----------------------------------------------------------------------
    def delete_chunks(
        self,
        chunk_ids: List[int],
        collection_id: Optional[int] = None
    ) -> int:
        """
        Remove chunks from the index.
        FAISS does not support removing individual vectors in this implementation.
        Therefore chunks are marked is_active=False in the DB and the index is rebuilt entirely.
        Returns the number of removed vectors (always 0 if not rebuilt).
        """
        if not chunk_ids or collection_id is None:
            return 0

        # Mark chunks as inactive
        DocumentChunk.objects.filter(id__in=chunk_ids).update(is_active=False)

        # Rebuild the index completely (a task can be launched)
        # Audit fix: rebuild_collection_index_task does not exist — the import
        # raised ImportError on every call
        from ..tasks import build_faiss_index_task
        build_faiss_index_task.delay(collection_id)

        logger.info(f"Marked {len(chunk_ids)} chunks as inactive, rebuild scheduled")
        return len(chunk_ids)  # for compatibility

    def remove_stale_vectors(self, collection_id: int, stale_chunk_ids: List[int]) -> int:
        """
        Remove vectors of deactivated chunks from the index.

        Audit fix: deactivated chunks kept their vectors in FAISS forever —
        search returned stale duplicates until a full rebuild. For flat
        indexes the vectors are reconstructed and the index rebuilt in place;
        for ivf/hnsw the collection is rebuilt from active DB chunks.
        Returns the number of removed vectors (0 if nothing changed).
        """
        from ..models import DocumentCollection

        if not stale_chunk_ids or collection_id is None:
            return 0

        svc = self._get_faiss_service(collection_id)
        if svc.index is None or not svc.chunk_ids:
            return 0

        stale = set(stale_chunk_ids)
        keep = [(i, cid) for i, cid in enumerate(svc.chunk_ids) if cid not in stale]
        if len(keep) == len(svc.chunk_ids):
            return 0  # nothing stale present

        removed = len(svc.chunk_ids) - len(keep)
        index_type = svc._detect_index_type()

        if index_type != 'flat':
            # No cheap removal path — rebuild from active DB chunks
            logger.info(f"Stale vector removal: rebuilding {index_type} index "
                        f"for collection {collection_id} ({removed} stale)")
            self.build_index(collection_id)
            return removed

        dimension = svc.dimension
        vectors = [svc.index.reconstruct(i) for i, _ in keep]
        new_ids = [cid for _, cid in keep]

        svc.create_index(dimension=dimension, index_type='flat', metric=svc.metric)
        if vectors:
            svc.add_vectors(np.stack(vectors).astype('float32'), new_ids)
        svc.save_index(collection_id)
        self._index_cache[collection_id] = svc

        DocumentCollection.objects.filter(id=collection_id).update(
            total_vectors=svc.index.ntotal)
        logger.info(f"Removed {removed} stale vectors from flat index "
                    f"of collection {collection_id} (now {svc.index.ntotal})")
        return removed

    # ----------------------------------------------------------------------
    # Collection index management
    # ----------------------------------------------------------------------
    def build_index(
        self,
        collection_id: int,
        index_type: str = 'flat',
        force: bool = False
    ) -> bool:
        """
        Build an index for the collection from scratch.
        - Collects all chunks of the collection that have embeddings.
        - Creates a FAISS index.
        - Saves it to disk.
        """
        from ..models import DocumentChunk
        import numpy as np
        from django.utils import timezone

        # Fetch all chunks of the collection with embeddings
        chunks = DocumentChunk.objects.filter(
            document__collection_id=collection_id,
            has_embedding=True,
            is_active=True
        ).exclude(embedding_vector__isnull=True)

        if not chunks.exists():
            logger.warning(f"No chunks with embeddings for collection {collection_id}")
            return False

        # Determine the dimension
        sample_vec = chunks.first().embedding_vector
        dimension = len(sample_vec) if isinstance(sample_vec, list) else sample_vec.shape[0]

        # Create a new service
        svc = FAISSService()
        svc.create_index(dimension=dimension, index_type=index_type, metric=self._default_metric)

        # Add the vectors
        vectors = []
        chunk_ids = []
        for chunk in chunks.iterator():
            vec = np.array(chunk.embedding_vector, dtype='float32')
            vectors.append(vec)
            chunk_ids.append(chunk.id)
            if len(vectors) >= 1000:  # add in batches
                svc.add_vectors(np.stack(vectors), chunk_ids)
                vectors = []
                chunk_ids = []

        if vectors:
            svc.add_vectors(np.stack(vectors), chunk_ids)

        # Save
        svc.save_index(collection_id)

        # Update the cache
        self._index_cache[collection_id] = svc

        # Update the collection fields
        from ..models import DocumentCollection
        now_iso = timezone.now().isoformat()
        DocumentCollection.objects.filter(id=collection_id).update(
            vector_index_created=True,
            total_vectors=svc.index.ntotal,
            vector_dimension=dimension,
            index_metadata={
                'index_type': index_type,
                'metric': self._default_metric,
                'built_at': now_iso,
                # Audit fix: update_index reads 'last_update' — it used to be
                # missing after build, so the first update re-added EVERY chunk
                # (index doubling). Same timestamp = only chunks created after
                # the build get indexed.
                'last_update': now_iso,
            }
        )

        logger.info(f"✅ Index built for collection {collection_id}: {svc.index.ntotal} vectors")
        return True

    def update_index(self, collection_id: int) -> int:
        """
        Update the index: add new chunks created after the last indexing.
        """
        from ..models import DocumentChunk, DocumentCollection

        collection = DocumentCollection.objects.filter(id=collection_id).first()
        if not collection or not collection.vector_index_created:
            logger.warning(f"Collection {collection_id} has no index, cannot update")
            return 0

        # Determine the last update date from metadata
        last_update = collection.index_metadata.get('last_update')
        if last_update:
            from datetime import datetime
            if isinstance(last_update, str):
                try:
                    last_update = datetime.fromisoformat(last_update)
                except ValueError:
                    last_update = None

        # New chunks with embeddings
        query = DocumentChunk.objects.filter(
            document__collection_id=collection_id,
            has_embedding=True,
            is_active=True
        ).exclude(embedding_vector__isnull=True)

        if last_update:
            query = query.filter(created_at__gt=last_update)

        new_chunks = list(query)
        if not new_chunks:
            return 0

        # Add to the index
        added = self.index_chunks(new_chunks, collection_id)

        # Update the metadata
        collection.total_vectors = self._get_faiss_service(collection_id).index.ntotal
        collection.index_metadata['last_update'] = timezone.now().isoformat()
        collection.index_metadata['last_update_count'] = added
        collection.save(update_fields=['total_vectors', 'index_metadata'])

        logger.info(f"Updated index for collection {collection_id}: added {added} vectors")
        return added

    # ----------------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------------
    def get_statistics(self, collection_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get vector store statistics.
        """
        if collection_id is None:
            # Aggregate statistics across all collections
            from ..models import DocumentCollection
            collections = DocumentCollection.objects.filter(is_active=True)
            return {
                'total_collections': collections.count(),
                'collections_with_index': collections.filter(vector_index_created=True).count(),
                'total_indexed_vectors': sum(c.total_vectors for c in collections),
            }
        else:
            svc = self._get_faiss_service(collection_id)
            stats = svc.get_index_stats(collection_id)
            stats['collection_id'] = collection_id
            return stats

    # ----------------------------------------------------------------------
    # Cache cleanup
    # ----------------------------------------------------------------------
    def clear_cache(self):
        """Clear the cache of loaded indexes."""
        self._index_cache.clear()
        logger.info("Vector store cache cleared")