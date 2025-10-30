"""
Setup Validation Script
Validates that the development environment is correctly configured for Week 2.

Run this before starting code review to catch any issues early.

Usage:
    python scripts/00_validate_setup.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SetupValidator:
    """Validates development environment setup."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def print_header(self):
        """Print header."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║         CODE INTELLIGENCE ASSISTANT - SETUP VALIDATOR           ║
║                    Week 1 → Week 2 Readiness                    ║
╚══════════════════════════════════════════════════════════════════╝
        """)
    
    def check_python_version(self) -> bool:
        """Check Python version."""
        print("\n🐍 Checking Python version...")
        print("-" * 70)
        
        version = sys.version_info
        current = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 10:
            print(f"✅ Python {current} (>= 3.10 required)")
            return True
        else:
            print(f"❌ Python {current} (>= 3.10 required)")
            print("   Please upgrade Python")
            return False
    
    def check_dependencies(self) -> bool:
        """Check required dependencies."""
        print("\n📦 Checking dependencies...")
        print("-" * 70)
        
        required = [
            ('dotenv', 'python-dotenv'),
            ('github', 'PyGithub'),
            ('sentence_transformers', 'sentence-transformers'),
            ('numpy', 'numpy'),
            ('tqdm', 'tqdm'),
            ('requests', 'requests'),
        ]
        
        all_installed = True
        
        for module_name, package_name in required:
            try:
                importlib.import_module(module_name)
                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name} - NOT INSTALLED")
                all_installed = False
        
        if not all_installed:
            print("\n💡 Install missing packages:")
            print("   pip install -r requirements.txt")
        
        return all_installed
    
    def check_environment_variables(self) -> bool:
        """Check environment variables."""
        print("\n🔐 Checking environment configuration...")
        print("-" * 70)
        
        from dotenv import load_dotenv
        load_dotenv()
        
        # Week 1 variables
        week1_vars = {
            'GITHUB_TOKEN': 'Required for data collection',
            'TARGET_REPOS': 'Target repositories',
        }
        
        # Week 2 variables
        week2_vars = {
            'LLM_PROVIDER': 'LLM provider (openrouter/ollama)',
            'OPENROUTER_API_KEY': 'OpenRouter API key (if using OpenRouter)',
            'LLM_MODEL': 'LLM model name',
        }
        
        all_ok = True
        
        print("\n📝 Week 1 Configuration:")
        for var, description in week1_vars.items():
            value = os.getenv(var)
            if value:
                masked_value = value[:10] + "..." if len(value) > 10 else value
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"⚠️  {var}: NOT SET ({description})")
                if var == 'GITHUB_TOKEN':
                    all_ok = False
        
        print("\n📝 Week 2 Configuration:")
        llm_provider = os.getenv('LLM_PROVIDER', 'openrouter')
        
        for var, description in week2_vars.items():
            value = os.getenv(var)
            
            # OpenRouter key only required if using OpenRouter
            if var == 'OPENROUTER_API_KEY' and llm_provider != 'openrouter':
                print(f"⏭️  {var}: SKIPPED (using {llm_provider})")
                continue
            
            if value:
                masked_value = value[:10] + "..." if len(value) > 10 else value
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"⚠️  {var}: NOT SET ({description})")
                self.warnings.append(f"Set {var} for Week 2")
        
        if not all_ok:
            print("\n💡 Fix:")
            print("   1. Copy .env.example to .env")
            print("   2. Fill in required values")
        
        return all_ok
    
    def check_week1_data(self) -> Tuple[bool, dict]:
        """Check Week 1 data artifacts."""
        print("\n📊 Checking Week 1 data artifacts...")
        print("-" * 70)
        
        checks = {
            'raw_data': Path('data/raw'),
            'processed_data': Path('data/processed/reviews.jsonl'),
            'embeddings': Path('data/embeddings/embeddings.npy'),
            'metadata': Path('data/embeddings/metadata.pkl'),
            'index': Path('data/embeddings/index.pkl'),
        }
        
        results = {}
        all_ok = True
        
        for name, path in checks.items():
            exists = path.exists()
            results[name] = exists
            
            if exists:
                if path.is_file():
                    size = path.stat().st_size
                    size_mb = size / 1024 / 1024
                    print(f"✅ {name}: {path} ({size_mb:.2f} MB)")
                else:
                    count = len(list(path.glob('**/*')))
                    print(f"✅ {name}: {path} ({count} files)")
            else:
                print(f"❌ {name}: {path} - NOT FOUND")
                all_ok = False
        
        if not all_ok:
            print("\n💡 Missing Week 1 data. Run:")
            if not results.get('raw_data'):
                print("   python scripts/01_collect_github_data.py")
            if not results.get('processed_data'):
                print("   python scripts/02_process_reviews.py")
            if not results.get('embeddings'):
                print("   python scripts/03_create_embeddings.py")
        
        return all_ok, results
    
    def check_week1_scripts(self) -> bool:
        """Check Week 1 scripts exist."""
        print("\n📄 Checking Week 1 scripts...")
        print("-" * 70)
        
        scripts = [
            '01_collect_github_data.py',
            '02_process_reviews.py',
            '03_create_embeddings.py',
            '04_test_retrieval.py',
        ]
        
        all_exist = True
        
        for script in scripts:
            path = Path(f'scripts/{script}')
            if path.exists():
                print(f"✅ {script}")
            else:
                print(f"❌ {script} - NOT FOUND")
                all_exist = False
        
        return all_exist
    
    def test_embeddings_loading(self) -> bool:
        """Test loading embeddings."""
        print("\n🧪 Testing embeddings loading...")
        print("-" * 70)
        
        try:
            import numpy as np
            import pickle
            
            embeddings_file = Path('data/embeddings/embeddings.npy')
            metadata_file = Path('data/embeddings/metadata.pkl')
            
            if not embeddings_file.exists():
                print("❌ Embeddings file not found")
                return False
            
            # Load embeddings
            embeddings = np.load(embeddings_file)
            print(f"✅ Loaded embeddings: {embeddings.shape}")
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✅ Loaded metadata: {len(metadata)} items")
            
            # Validate consistency
            if len(embeddings) == len(metadata):
                print(f"✅ Embeddings and metadata consistent")
                return True
            else:
                print(f"❌ Mismatch: {len(embeddings)} embeddings, {len(metadata)} metadata")
                return False
                
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def test_retrieval(self) -> bool:
        """Test basic retrieval."""
        print("\n🔍 Testing retrieval pipeline...")
        print("-" * 70)
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            import pickle
            
            # Load model
            print("Loading model...")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load embeddings
            embeddings = np.load('data/embeddings/embeddings.npy')
            with open('data/embeddings/metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Test query
            query = "def process_data(): pass"
            query_embedding = model.encode([query], convert_to_numpy=True)[0]
            
            # Compute similarities
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, query_norm)
            
            # Get top result
            top_idx = np.argmax(similarities)
            top_similarity = similarities[top_idx]
            
            print(f"✅ Retrieval working!")
            print(f"   Top similarity: {top_similarity:.3f}")
            print(f"   Top result: [{metadata[top_idx]['repo']}]")
            
            return True
            
        except Exception as e:
            print(f"❌ Retrieval test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_llm_providers(self) -> dict:
        """Check LLM provider availability."""
        print("\n🤖 Checking LLM providers...")
        print("-" * 70)
        
        results = {}
        
        # Check OpenRouter
        print("\n1️⃣  OpenRouter:")
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            print(f"   ✅ API key configured")
            results['openrouter'] = True
        else:
            print(f"   ⚠️  API key not set")
            print(f"   💡 Get key from: https://openrouter.ai/keys")
            results['openrouter'] = False
        
        # Check Ollama
        print("\n2️⃣  Ollama:")
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.ok:
                models = response.json().get('models', [])
                print(f"   ✅ Ollama running")
                if models:
                    print(f"   📦 Models: {', '.join(m['name'] for m in models[:3])}")
                else:
                    print(f"   ⚠️  No models installed")
                    print(f"   💡 Run: ollama pull llama3.1:8b")
                results['ollama'] = True
            else:
                print(f"   ⚠️  Ollama not responding")
                results['ollama'] = False
        except Exception:
            print(f"   ⚠️  Ollama not running")
            print(f"   💡 Install from: https://ollama.ai")
            results['ollama'] = False
        
        # Summary
        print("\n📊 LLM Provider Summary:")
        if results.get('openrouter') or results.get('ollama'):
            print("   ✅ At least one provider available")
        else:
            print("   ❌ No LLM provider configured")
            print("   💡 Configure OpenRouter OR install Ollama for Week 2")
        
        return results
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        self.print_header()
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Environment Variables", self.check_environment_variables),
            ("Week 1 Scripts", self.check_week1_scripts),
        ]
        
        results = {}
        
        for name, check_func in checks:
            try:
                result = check_func()
                results[name] = result
                
                if result:
                    self.checks_passed += 1
                else:
                    self.checks_failed += 1
                    
            except Exception as e:
                logger.error(f"Check '{name}' failed with exception: {e}")
                results[name] = False
                self.checks_failed += 1
        
        # Data checks
        data_ok, data_results = self.check_week1_data()
        results["Week 1 Data"] = data_ok
        if data_ok:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
        
        # Advanced checks (only if basics pass)
        if data_ok:
            print("\n" + "="*70)
            print("ADVANCED VALIDATION")
            print("="*70)
            
            embeddings_ok = self.test_embeddings_loading()
            results["Embeddings Loading"] = embeddings_ok
            if embeddings_ok:
                self.checks_passed += 1
            else:
                self.checks_failed += 1
            
            if embeddings_ok:
                retrieval_ok = self.test_retrieval()
                results["Retrieval Pipeline"] = retrieval_ok
                if retrieval_ok:
                    self.checks_passed += 1
                else:
                    self.checks_failed += 1
        
        # LLM provider check (informational)
        llm_results = self.check_llm_providers()
        
        # Print summary
        self.print_summary(results, llm_results)
        
        return self.checks_failed == 0
    
    def print_summary(self, results: dict, llm_results: dict):
        """Print validation summary."""
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        
        # Core checks
        print(f"\n✅ Passed: {self.checks_passed}")
        print(f"❌ Failed: {self.checks_failed}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        # Overall status
        print("\n" + "="*70)
        
        if self.checks_failed == 0:
            print("🎉 READY FOR WEEK 2!")
            print("="*70)
            print("\n✨ Your environment is properly configured.")
            print("\n📝 Next steps:")
            print("   1. Review Week 2 implementation guide")
            print("   2. Configure LLM provider (OpenRouter or Ollama)")
            print("   3. Run: python scripts/05_rag_pipeline.py")
            
            if not llm_results.get('openrouter') and not llm_results.get('ollama'):
                print("\n⚠️  NOTE: No LLM provider configured yet")
                print("   This is OK - you can configure it when starting Week 2")
        
        elif self.checks_failed <= 2 and all(
            results.get(k, False) for k in ['Python Version', 'Dependencies', 'Week 1 Data']
        ):
            print("⚠️  MOSTLY READY - MINOR ISSUES")
            print("="*70)
            print("\n✨ Core functionality is ready.")
            print("⚠️  Some optional checks failed - review above.")
            
        else:
            print("❌ NOT READY - SETUP REQUIRED")
            print("="*70)
            print("\n🔧 Please fix the issues above before proceeding.")
            print("\n💡 Common fixes:")
            print("   1. Run Week 1 scripts to generate data")
            print("   2. Install missing dependencies")
            print("   3. Configure .env file")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main execution."""
    validator = SetupValidator()
    
    try:
        success = validator.run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()