"""
Hybrid Code Reviewer - OpenRouter + Ollama
Uses Ollama for fast local screening, OpenRouter as cloud backup

Save this as: scripts/hybrid_reviewer.py
"""

import ollama
import json
import sys
import time
import os
import requests
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠ RAG not available: {e}")
    RAG_AVAILABLE = False


class HybridReviewer:
    """Hybrid reviewer using Ollama (local) + OpenRouter (cloud)."""
    
    def __init__(self, use_ollama=True, use_openrouter=True):
        self.use_ollama = use_ollama
        self.use_openrouter = use_openrouter
        
        # Get config
        self.ollama_model = 'qwen2.5-coder:7b'
        self.openrouter_model = os.getenv('LLM_MODEL', 'meta-llama/llama-3.1-8b-instruct')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        print(f"\n{'='*70}")
        print("HYBRID REVIEWER INITIALIZATION")
        print(f"{'='*70}")
        
        # Init Ollama
        if self.use_ollama:
            try:
                self.ollama_client = ollama.Client()
                models_list = self.ollama_client.list()
                
                # Handle different response formats
                if isinstance(models_list, dict):
                    models = models_list.get('models', [])
                else:
                    models = models_list
                
                # Extract model names safely
                available = []
                for m in models:
                    if isinstance(m, dict):
                        # Try different key names
                        name = m.get('name') or m.get('model') or m.get('id', '')
                        if name:
                            available.append(name)
                    elif isinstance(m, str):
                        available.append(m)
                
                print(f"📋 Ollama models found: {available}")
                
                if not available:
                    print(f"⚠ No models found, pulling {self.ollama_model}...")
                    self.ollama_client.pull(self.ollama_model)
                    available.append(self.ollama_model)
                elif not any(self.ollama_model in m for m in available):
                    print(f"⚠ Model {self.ollama_model} not found, pulling...")
                    self.ollama_client.pull(self.ollama_model)
                    available.append(self.ollama_model)
                
                print(f"✓ Ollama ready: {self.ollama_model}")
            except Exception as e:
                import traceback
                print(f"⚠ Ollama failed: {e}")
                print(f"   Debug info: {traceback.format_exc()}")
                self.use_ollama = False
        
        # Init OpenRouter
        if self.use_openrouter:
            if not self.openrouter_api_key:
                print(f"⚠ No OpenRouter key")
                self.use_openrouter = False
            else:
                print(f"✓ OpenRouter: {self.openrouter_model}")
        
        # Init RAG
        if RAG_AVAILABLE:
            try:
                self.rag = RAGPipeline(top_k=3, rerank_top_n=2)
                print(f"✓ RAG loaded")
            except:
                self.rag = None
        else:
            self.rag = None
        
        print(f"{'='*70}\n")
    
    def build_prompt(self, code, detailed=False):
        if detailed:
            prompt = f"""You are a code review expert. Analyze this Python code and provide ONLY a JSON response.

Code to review:
```python
{code}
```

CRITICAL: Respond with ONLY valid JSON, no other text. Use this exact format:

{{
  "summary": "Brief overview of code quality and main issues",
  "issues": [
    {{
      "category": "BUG|SECURITY|PERFORMANCE|STYLE|DOCUMENTATION|MAINTAINABILITY",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "title": "Issue title",
      "description": "Detailed description",
      "location": "Function/line reference",
      "suggestion": "How to fix"
    }}
  ],
  "suggestions": ["General improvement 1", "General improvement 2"],
  "overall_quality": "EXCELLENT|GOOD|FAIR|POOR"
}}

Remember: ONLY return valid JSON, nothing else."""
        else:
            prompt = f"""Quick code analysis. Respond with ONLY valid JSON.

Code:
```python
{code}
```

Return ONLY this JSON format:
{{
  "severity": "CRITICAL|HIGH|MEDIUM|LOW|NONE",
  "main_issues": ["issue 1", "issue 2"],
  "needs_detailed_review": true
}}

ONLY JSON, no explanation."""
        
        return prompt
    
    def review_with_ollama(self, code, quick=True):
        print(f"⚡ Ollama {'screening' if quick else 'review'}...")
        start = time.time()
        
        try:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": self.build_prompt(code, not quick)}],
                options={"temperature": 0.1, "num_predict": 1000 if quick else 2000}
            )
            
            content = response['message']['content']
            duration = time.time() - start
            
            # Parse JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            result['_time'] = duration
            result['_model'] = f"ollama/{self.ollama_model}"
            
            print(f"✓ Ollama: {duration:.2f}s")
            return result
        except Exception as e:
            print(f"⚠ Ollama error: {e}")
            return {"severity": "MEDIUM", "needs_detailed_review": True, "_time": time.time()-start}
    
    def review_with_openrouter(self, code):
        print(f"🌐 OpenRouter review...")
        start = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": self.openrouter_model,
                "messages": [
                    {
                        "role": "user", 
                        "content": self.build_prompt(code, True)
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/code-review-assistant",
                "X-Title": "Code Review Assistant"
            }
            
            print(f"   Sending to: {self.openrouter_model}")
            print(f"   Prompt length: {len(payload['messages'][0]['content'])} chars")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            
            duration = time.time() - start
            
            # Debug response status
            print(f"   Response status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text[:500]
                print(f"⚠ OpenRouter error response: {error_text}")
                
                # Try to parse error JSON
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', error_text)
                    print(f"   Error: {error_msg}")
                except:
                    pass
                
                raise Exception(f"HTTP {response.status_code}: {error_text}")
            
            # Check response content
            if not response.text or len(response.text) < 10:
                raise Exception(f"Empty/invalid response (length: {len(response.text)})")
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"⚠ Cannot parse JSON response")
                print(f"   Response preview: {response.text[:200]}")
                raise Exception(f"Invalid JSON: {e}")
            
            # Check for API errors
            if 'error' in data:
                error_obj = data['error']
                if isinstance(error_obj, dict):
                    error_msg = error_obj.get('message', str(error_obj))
                else:
                    error_msg = str(error_obj)
                raise Exception(f"API error: {error_msg}")
            
            # Extract content
            if 'choices' not in data or not data['choices']:
                raise Exception(f"No choices in response: {data}")
            
            content = data['choices'][0]['message']['content']
            print(f"   Response length: {len(content)} chars")
            
            # Parse review JSON from content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                print(f"⚠ Cannot parse review JSON")
                print(f"   Content preview: {content[:200]}")
                # Return raw content as summary
                result = {
                    "summary": content[:500],
                    "issues": [],
                    "suggestions": [content[:200]],
                    "overall_quality": "UNKNOWN",
                    "raw_content": content
                }
            
            result['_time'] = duration
            result['_model'] = f"openrouter/{self.openrouter_model}"
            
            print(f"✓ OpenRouter: {duration:.2f}s")
            return result
            
        except requests.exceptions.Timeout:
            print(f"⚠ OpenRouter timeout after 45s")
            return {
                "summary": "Request timed out after 45s",
                "issues": [],
                "suggestions": [],
                "_time": 45,
                "_error": "timeout"
            }
        except requests.exceptions.RequestException as e:
            print(f"⚠ OpenRouter network error: {e}")
            return {
                "summary": f"Network error: {str(e)[:200]}",
                "issues": [],
                "suggestions": [],
                "_time": time.time() - start,
                "_error": str(e)
            }
        except Exception as e:
            print(f"⚠ OpenRouter error: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()[:300]}")
            return {
                "summary": f"Error: {str(e)[:200]}",
                "issues": [],
                "suggestions": [],
                "_time": time.time() - start,
                "_error": str(e)
            }
    
    def review_code(self, code, file_path=None):
        print(f"\n{'='*70}")
        print(f"🔍 Reviewing: {file_path or 'code'}")
        print(f"{'='*70}")
        
        total_start = time.time()
        strategy = "unknown"
        screening = None
        review = None
        
        # Strategy: Ollama screen → OpenRouter if needed
        if self.use_ollama:
            screening = self.review_with_ollama(code, quick=True)
            severity = screening.get('severity', 'MEDIUM')
            needs_detail = screening.get('needs_detailed_review', True)
            
            print(f"   Severity: {severity}")
            
            if severity in ['CRITICAL', 'HIGH'] or needs_detail:
                if self.use_openrouter:
                    strategy = "hybrid_detailed"
                    review = self.review_with_openrouter(code)
                else:
                    strategy = "ollama_detailed"
                    review = self.review_with_ollama(code, quick=False)
            else:
                strategy = "ollama_fast"
                review = {
                    "summary": f"Quick screen: {severity}",
                    "issues": [{"category": "GENERAL", "severity": severity, "title": i, "description": i, "location": "code", "suggestion": "review"} for i in screening.get('main_issues', [])],
                    "suggestions": screening.get('main_issues', []),
                    "overall_quality": "GOOD" if severity in ['LOW','NONE'] else "FAIR"
                }
        else:
            strategy = "openrouter_only"
            review = self.review_with_openrouter(code)
        
        total_time = time.time() - total_start
        
        result = {
            "file": file_path,
            "strategy": strategy,
            "review": review or {},
            "performance": {
                "total_time": total_time,
                "screening_time": screening.get('_time', 0) if screening else 0,
                "detailed_time": review.get('_time', 0) if review else 0,
                "model_used": review.get('_model', 'unknown') if review else 'none',
                "cost": 0.0
            }
        }
        
        if screening:
            result["screening"] = screening
        
        issues = len(result['review'].get('issues', []))
        print(f"\n✓ Complete: {issues} issues, {total_time:.2f}s, strategy: {strategy}")
        print(f"{'='*70}\n")
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Reviewer')
    parser.add_argument('--file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-ollama', action='store_true')
    parser.add_argument('--no-openrouter', action='store_true')
    
    args = parser.parse_args()
    
    # Read file
    code_file = Path(args.file)
    if not code_file.exists():
        print(f"❌ File not found: {args.file}")
        return 1
    
    code = code_file.read_text(encoding='utf-8')
    
    # Create reviewer
    try:
        reviewer = HybridReviewer(
            use_ollama=not args.no_ollama,
            use_openrouter=not args.no_openrouter
        )
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return 1
    
    # Review
    result = reviewer.review_code(code, str(code_file))
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 Saved: {output_path}")
    
    # Summary
    review = result.get('review', {})
    print("\n" + "="*70)
    print("REVIEW SUMMARY")
    print("="*70)
    print(f"Strategy: {result.get('strategy')}")
    print(f"Issues: {len(review.get('issues', []))}")
    print(f"Time: {result['performance']['total_time']:.2f}s")
    print(f"Cost: $0.00")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())