"""
Retrieval Testing Script - Updated for NumPy Embeddings
Tests the quality of vector similarity search using saved NumPy embeddings.

Compatible with the Pure NumPy solution (03_create_embeddings.py)
"""

import json
from pathlib import Path
from typing import List, Dict
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RetrievalTester:
    """Tests retrieval quality of the embedded code reviews (NumPy version)."""
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        self.embeddings_dir = Path('data/embeddings')
        self.model_name = model_name
        self.model = None
        
        # Will be loaded from disk
        self.embeddings = None
        self.metadatas = None
        self.id_to_idx = None
        
        # Test results
        self.test_results = []
    
    def setup(self):
        """Load model and embeddings from disk."""
        logger.info("="*60)
        logger.info("RETRIEVAL TESTING SETUP")
        logger.info("="*60)
        
        # Load model
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("✅ Model loaded")
        
        # Load embeddings from NumPy files
        logger.info(f"\nLoading embeddings from: {self.embeddings_dir}")
        
        try:
            # Load embeddings
            embeddings_file = self.embeddings_dir / 'embeddings.npy'
            self.embeddings = np.load(embeddings_file)
            logger.info(f"✅ Loaded embeddings: {self.embeddings.shape}")
            
            # Load metadata
            import pickle
            metadata_file = self.embeddings_dir / 'metadata.pkl'
            with open(metadata_file, 'rb') as f:
                self.metadatas = pickle.load(f)
            logger.info(f"✅ Loaded metadata: {len(self.metadatas)} items")
            
            # Load index
            index_file = self.embeddings_dir / 'index.pkl'
            with open(index_file, 'rb') as f:
                self.id_to_idx = pickle.load(f)
            logger.info(f"✅ Loaded ID index")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            logger.info("\nPlease run embeddings generation first:")
            logger.info("   python scripts/03_create_embeddings.py")
            raise
        
        logger.info("="*60)
    
    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and all embeddings.
        
        Args:
            query_embedding: Query vector of shape (384,)
        
        Returns:
            similarities: Array of shape (n_samples,)
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize embeddings
        embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        # Compute dot product
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def test_query(
        self,
        query: str,
        n_results: int = 5,
        show_details: bool = True
    ) -> Dict:
        """
        Test a single query and return results.
        
        Args:
            query: Search query
            n_results: Number of results to return
            show_details: Whether to print detailed results
        
        Returns:
            Dictionary with query results and analysis
        """
        if show_details:
            logger.info(f"\n🔍 Query: '{query}'")
            logger.info("-" * 60)
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            
            # Compute similarities
            similarities = self.cosine_similarity(query_embedding)
            
            # Get top N indices
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            # Get results
            results = []
            for idx in top_indices:
                meta = self.metadatas[idx].copy()
                meta['similarity'] = float(similarities[idx])
                results.append(meta)
            
            if show_details:
                for i, result in enumerate(results, 1):
                    logger.info(f"\n{i}. [{result['repo']}] (similarity: {result['similarity']:.3f})")
                    logger.info(f"   Reviewer: {result['reviewer']}")
                    logger.info(f"   File: {result['file']}")
                    logger.info(f"   Review: {result['comment'][:100]}...")
            
            return {
                'query': query,
                'num_results': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {'query': query, 'error': str(e)}
    
    def run_test_suite(self):
        """Run comprehensive test suite with various query types."""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE TEST SUITE")
        logger.info("="*60)
        
        test_queries = {
            "Error Handling": [
                "How to handle errors in async functions?",
                "Missing exception handling",
                "Error handling best practices"
            ],
            "Type Safety": [
                "Type validation for parameters",
                "Missing type hints",
                "Type checking improvements"
            ],
            "Performance": [
                "Performance optimization needed",
                "Slow database queries",
                "Memory usage concerns"
            ],
            "Documentation": [
                "Documentation is incomplete",
                "Missing docstrings",
                "Update documentation for this feature"
            ],
            "API Design": [
                "API endpoint design issues",
                "REST API best practices",
                "Request validation needed"
            ],
            "Testing": [
                "Need more test coverage",
                "Missing edge case tests",
                "Test this functionality"
            ],
            "Code Quality": [
                "This code is too complex",
                "Refactor this function",
                "Code duplication here"
            ]
        }
        
        category_results = {}
        
        for category, queries in test_queries.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {category}")
            logger.info(f"{'='*60}")
            
            category_results[category] = []
            
            for query in queries:
                result = self.test_query(query, n_results=3, show_details=True)
                category_results[category].append(result)
                self.test_results.append(result)
        
        return category_results
    
    def analyze_retrieval_quality(self):
        """Analyze the quality of retrieval results."""
        logger.info("\n" + "="*60)
        logger.info("RETRIEVAL QUALITY ANALYSIS")
        logger.info("="*60)
        
        # Analyze diversity of results
        all_repos = []
        all_reviewers = []
        all_similarities = []
        
        for test in self.test_results:
            if 'results' in test:
                for result in test['results']:
                    all_repos.append(result.get('repo', ''))
                    all_reviewers.append(result.get('reviewer', ''))
                    all_similarities.append(result.get('similarity', 0))
        
        repo_dist = Counter(all_repos)
        reviewer_dist = Counter(all_reviewers)
        
        logger.info("\n📊 Result Diversity:")
        logger.info(f"   Unique repos in results: {len(repo_dist)}")
        logger.info(f"   Unique reviewers in results: {len(reviewer_dist)}")
        
        logger.info("\n📈 Repository Distribution:")
        for repo, count in repo_dist.most_common():
            percentage = (count / len(all_repos)) * 100
            logger.info(f"   {repo}: {count} ({percentage:.1f}%)")
        
        logger.info("\n👥 Top Reviewers in Results:")
        for reviewer, count in reviewer_dist.most_common(5):
            percentage = (count / len(all_reviewers)) * 100
            logger.info(f"   {reviewer}: {count} ({percentage:.1f}%)")
        
        logger.info("\n🎯 Similarity Scores:")
        if all_similarities:
            avg_sim = np.mean(all_similarities)
            max_sim = np.max(all_similarities)
            min_sim = np.min(all_similarities)
            logger.info(f"   Average: {avg_sim:.3f}")
            logger.info(f"   Max: {max_sim:.3f}")
            logger.info(f"   Min: {min_sim:.3f}")
    
    def interactive_mode(self):
        """Interactive mode for testing custom queries."""
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE QUERY MODE")
        logger.info("="*60)
        logger.info("\nEnter queries to test retrieval (type 'quit' to exit)")
        logger.info("Examples:")
        logger.info("  - How to handle async errors?")
        logger.info("  - Type validation missing")
        logger.info("  - Performance optimization needed")
        logger.info("-" * 60)
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("Exiting interactive mode...")
                    break
                
                if not query:
                    continue
                
                self.test_query(query, n_results=5, show_details=True)
                
            except KeyboardInterrupt:
                logger.info("\n\nExiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def print_summary(self):
        """Print final summary."""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total queries tested: {len(self.test_results)}")
        
        successful = sum(1 for t in self.test_results if 'error' not in t)
        logger.info(f"Successful queries: {successful}/{len(self.test_results)}")
        
        logger.info("\n✅ Retrieval testing complete!")
        logger.info("="*60)


def main():
    """Main execution function."""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Code Intelligence Assistant - Retrieval Testing          ║
║  NumPy Embeddings Version                                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        tester = RetrievalTester()
        tester.setup()
        
        # Check if interactive mode requested
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            tester.interactive_mode()
        else:
            # Run test suite
            tester.run_test_suite()
            
            # Analyze quality
            tester.analyze_retrieval_quality()
            
            # Print summary
            tester.print_summary()
            
            # Offer interactive mode
            logger.info("\n💡 Tip: Run with --interactive flag for custom queries:")
            logger.info("   python scripts/04_test_retrieval.py --interactive")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()