"""
Embeddings Storage - PURE NUMPY SOLUTION (No ChromaDB at all!)
Bypasses ChromaDB completely - stores embeddings as numpy arrays

This is the MOST reliable solution for Windows.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class EmbeddingsGenerator:
    """Generates and stores embeddings - Pure NumPy version (no ChromaDB!)."""
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        self.processed_file = Path('data/processed/reviews.jsonl')
        self.embeddings_dir = Path('data/embeddings')
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files
        self.embeddings_file = self.embeddings_dir / 'embeddings.npy'
        self.metadata_file = self.embeddings_dir / 'metadata.pkl'
        self.index_file = self.embeddings_dir / 'index.pkl'
        
        self.model_name = model_name
        self.model = None
    
    def setup(self):
        """Setup sentence transformer model."""
        logger.info("="*60)
        logger.info("PURE NUMPY SOLUTION (No ChromaDB)")
        logger.info("="*60)
        
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"✅ Model loaded in {time.time() - start_time:.2f}s")
        logger.info("="*60)
    
    def load_examples(self) -> List[Dict]:
        """Load examples."""
        logger.info(f"\nLoading: {self.processed_file}")
        
        examples = []
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except:
                    pass
        
        logger.info(f"✅ Loaded {len(examples)} examples")
        return examples
    
    def create_embedding_text(self, example: Dict) -> str:
        """Create text for embedding."""
        parts = []
        
        if example.get('file_path'):
            parts.append(f"File: {example['file_path']}")
        
        code = example.get('code_snippet', '').strip()
        if code:
            parts.append(f"Code:\n{code[:500]}")
        
        review = example.get('review_comment', '').strip()
        if review:
            parts.append(f"Review: {review}")
        
        return "\n\n".join(parts)
    
    def generate_and_save_embeddings(self, examples: List[Dict]):
        """Generate embeddings and save directly to disk."""
        logger.info("\n" + "="*60)
        logger.info("GENERATING EMBEDDINGS")
        logger.info("="*60)
        
        texts = []
        metadatas = []
        ids = []
        
        logger.info("Preparing texts...")
        
        for i, example in enumerate(examples):
            text = self.create_embedding_text(example)
            texts.append(text)
            
            metadata = {
                'id': example.get('id', f"review_{i}"),
                'repo': str(example.get('repo_name', ''))[:50],
                'pr': int(example.get('pr_number', 0)),
                'file': str(example.get('file_path', ''))[:100],
                'reviewer': str(example.get('reviewer', ''))[:30],
                'comment': str(example.get('review_comment', ''))[:150],
                'text': text[:800]
            }
            
            metadatas.append(metadata)
            ids.append(metadata['id'])
        
        logger.info(f"✅ Prepared {len(texts)} texts")
        
        # Generate embeddings
        logger.info(f"\nGenerating embeddings...")
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"\n✅ Generated {len(embeddings)} embeddings in {time.time() - start_time:.2f}s")
        logger.info(f"   Shape: {embeddings.shape}")
        
        # Save to disk immediately (no ChromaDB!)
        logger.info("\n" + "="*60)
        logger.info("SAVING TO DISK (NumPy + Pickle)")
        logger.info("="*60)
        
        # Save embeddings as numpy array
        logger.info(f"Saving embeddings to: {self.embeddings_file}")
        np.save(self.embeddings_file, embeddings)
        logger.info(f"✅ Embeddings saved ({embeddings.nbytes / 1024 / 1024:.2f} MB)")
        
        # Save metadata
        logger.info(f"Saving metadata to: {self.metadata_file}")
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadatas, f)
        logger.info(f"✅ Metadata saved ({self.metadata_file.stat().st_size / 1024:.2f} KB)")
        
        # Save ID index
        logger.info(f"Saving ID index to: {self.index_file}")
        id_to_idx = {id_val: idx for idx, id_val in enumerate(ids)}
        with open(self.index_file, 'wb') as f:
            pickle.dump(id_to_idx, f)
        logger.info(f"✅ Index saved")
        
        logger.info("\n" + "="*60)
        logger.info("FINAL STATISTICS")
        logger.info("="*60)
        logger.info(f"Total examples:       {len(examples)}")
        logger.info(f"Embeddings generated: {len(embeddings)}")
        logger.info(f"Embeddings saved:     {len(embeddings)}")
        logger.info(f"\n📁 Files created:")
        logger.info(f"   {self.embeddings_file}")
        logger.info(f"   {self.metadata_file}")
        logger.info(f"   {self.index_file}")
        logger.info("\n✅ SUCCESS! Ready for RAG pipeline!")
        logger.info("="*60)


def load_embeddings() -> Tuple[np.ndarray, List[Dict], Dict[str, int]]:
    """
    Load embeddings from disk.
    
    Returns:
        embeddings: numpy array of shape (n_samples, 384)
        metadatas: list of metadata dicts
        id_to_idx: dict mapping IDs to indices
    """
    embeddings_dir = Path('data/embeddings')
    
    logger.info("Loading embeddings from disk...")
    
    # Load embeddings
    embeddings_file = embeddings_dir / 'embeddings.npy'
    embeddings = np.load(embeddings_file)
    logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
    
    # Load metadata
    metadata_file = embeddings_dir / 'metadata.pkl'
    with open(metadata_file, 'rb') as f:
        metadatas = pickle.load(f)
    logger.info(f"✅ Loaded metadata: {len(metadatas)} items")
    
    # Load index
    index_file = embeddings_dir / 'index.pkl'
    with open(index_file, 'rb') as f:
        id_to_idx = pickle.load(f)
    logger.info(f"✅ Loaded index")
    
    return embeddings, metadatas, id_to_idx


def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and all embeddings.
    
    Args:
        query_embedding: shape (384,)
        embeddings: shape (n_samples, 384)
    
    Returns:
        similarities: shape (n_samples,)
    """
    # Normalize query
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute dot product
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities


def query_similar_reviews(query_text: str, n_results: int = 5) -> List[Dict]:
    """
    Find similar code reviews for a query.
    
    Args:
        query_text: Code or query to search for
        n_results: Number of results to return
    
    Returns:
        List of metadata dicts for top similar reviews
    """
    # Load embeddings
    embeddings, metadatas, id_to_idx = load_embeddings()
    
    # Load model
    logger.info("Loading model for query encoding...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Encode query
    logger.info(f"Encoding query: '{query_text[:50]}...'")
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
    
    # Compute similarities
    logger.info("Computing similarities...")
    similarities = cosine_similarity(query_embedding, embeddings)
    
    # Get top N indices
    top_indices = np.argsort(similarities)[-n_results:][::-1]
    
    # Get results
    results = []
    for idx in top_indices:
        meta = metadatas[idx].copy()
        meta['similarity'] = float(similarities[idx])
        results.append(meta)
    
    return results


def test_retrieval():
    """Test retrieval with sample queries."""
    logger.info("\n" + "="*60)
    logger.info("TESTING RETRIEVAL")
    logger.info("="*60)
    
    queries = [
        "async def fetch_data():\n    return await db.query()",
        "def validate_email(email: str):\n    if '@' not in email:\n        raise ValueError()",
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n🔍 Query {i}: '{query[:50]}...'")
        
        results = query_similar_reviews(query, n_results=3)
        
        logger.info("\nTop 3 Similar Reviews:")
        for j, result in enumerate(results, 1):
            logger.info(f"   {j}. [{result['repo']}] (similarity: {result['similarity']:.3f})")
            logger.info(f"      Reviewer: {result['reviewer']}")
            logger.info(f"      Comment: {result['comment'][:80]}...")


def main():
    """Main execution."""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Code Intelligence Assistant - Embeddings Generation      ║
║  PURE NUMPY SOLUTION (No ChromaDB - Most Reliable!)       ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        generator = EmbeddingsGenerator()
        generator.setup()
        
        examples = generator.load_examples()
        
        if not examples:
            logger.error("❌ No examples found!")
            return
        
        generator.generate_and_save_embeddings(examples)
        
        # Test retrieval
        logger.info("\nTesting retrieval...")
        test_retrieval()
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n" + "="*60)


if __name__ == '__main__':
    main()