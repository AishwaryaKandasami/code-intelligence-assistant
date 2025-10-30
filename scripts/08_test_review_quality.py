"""
Review Quality Testing Suite
Comprehensive testing of the code review pipeline.

Tests:
- Retrieval quality
- Review accuracy
- Response time
- Cost tracking
- Edge cases
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
import logging
from dataclasses import dataclass
from collections import defaultdict

# Fix imports - add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from local scripts (use actual filenames without .py)
try:
    # Try importing as modules
    import importlib.util
    
    # Import 07_end_to_end_review
    spec = importlib.util.spec_from_file_location(
        "end_to_end_review",
        Path(__file__).parent / "07_endreview.py"
    )
    end_to_end = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(end_to_end)
    CodeReviewAssistant = end_to_end.CodeReviewAssistant
    
except Exception as e:
    print(f"Error importing: {e}")
    print("\nPlease ensure 07_end_to_end_review.py exists in scripts/ directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case definition."""
    name: str
    code: str
    expected_issues: List[str]  # Expected issue categories
    description: str


class ReviewQualityTester:
    """Test suite for review quality."""
    
    def __init__(self):
        self.assistant = CodeReviewAssistant(save_history=False)
        self.test_results = []
        
        # Define test cases
        self.test_cases = [
            TestCase(
                name="SQL Injection",
                code='''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
''',
                expected_issues=['security', 'bug'],
                description="Should detect SQL injection vulnerability"
            ),
            
            TestCase(
                name="Missing Error Handling",
                code='''
async def fetch_data(url):
    response = await client.get(url)
    data = response.json()
    return data['result']
''',
                expected_issues=['bug', 'error handling'],
                description="Should suggest error handling"
            ),
            
            TestCase(
                name="Missing Type Hints",
                code='''
def calculate_total(items, tax_rate):
    subtotal = sum(item['price'] * item['qty'] for item in items)
    return subtotal * (1 + tax_rate)
''',
                expected_issues=['style', 'type'],
                description="Should recommend type hints"
            ),
            
            TestCase(
                name="Performance Issue",
                code='''
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates
''',
                expected_issues=['performance'],
                description="Should suggest set-based approach"
            ),
            
            TestCase(
                name="Complex Function",
                code='''
def process_order(order, user, inventory, shipping):
    if order['status'] == 'pending':
        if user['verified']:
            for item in order['items']:
                if inventory[item['id']] >= item['qty']:
                    inventory[item['id']] -= item['qty']
                    if shipping['method'] == 'express':
                        item['ship_date'] = datetime.now() + timedelta(days=1)
                    else:
                        item['ship_date'] = datetime.now() + timedelta(days=5)
                else:
                    return {'error': 'insufficient inventory'}
            order['status'] = 'processing'
            return order
        else:
            return {'error': 'user not verified'}
    return {'error': 'invalid status'}
''',
                expected_issues=['maintainability', 'complex'],
                description="Should suggest refactoring for complexity"
            ),
            
            TestCase(
                name="Missing Documentation",
                code='''
def calculate_discount(price, customer_type, quantity):
    if customer_type == 'premium':
        base_discount = 0.15
    elif customer_type == 'regular':
        base_discount = 0.05
    else:
        base_discount = 0
    
    if quantity > 100:
        base_discount += 0.10
    elif quantity > 50:
        base_discount += 0.05
    
    return price * (1 - base_discount)
''',
                expected_issues=['documentation'],
                description="Should recommend docstrings"
            ),
            
            TestCase(
                name="Good Code (Baseline)",
                code='''
from typing import List, Optional
from decimal import Decimal

def calculate_total_price(
    items: List[Dict[str, Decimal]], 
    tax_rate: Decimal,
    discount: Optional[Decimal] = None
) -> Decimal:
    """
    Calculate total price with tax and optional discount.
    
    Args:
        items: List of items with 'price' and 'quantity'
        tax_rate: Tax rate as decimal (e.g., 0.08 for 8%)
        discount: Optional discount rate
        
    Returns:
        Total price including tax
    """
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    
    if discount:
        subtotal *= (1 - discount)
    
    total = subtotal * (1 + tax_rate)
    return total
''',
                expected_issues=[],  # Should have minimal issues
                description="Well-written code - baseline test"
            )
        ]
    
    def run_test_case(self, test: TestCase) -> Dict:
        """Run single test case."""
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST: {test.name}")
        logger.info(f"Description: {test.description}")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Get review
            result = self.assistant.review_code(test.code)
            
            latency = time.time() - start_time
            
            # Analyze review
            review_text = result['review']['raw_content'].lower()
            
            # Check if expected issues were mentioned
            issues_found = []
            for expected in test.expected_issues:
                if expected.lower() in review_text:
                    issues_found.append(expected)
            
            # Count actual issues identified
            num_issues = len(result['review']['issues'])
            num_suggestions = len(result['review']['suggestions'])
            
            # Success criteria
            if test.expected_issues:
                # Should find issues
                success = len(issues_found) > 0
            else:
                # Good code - should have few issues
                success = num_issues <= 2
            
            test_result = {
                'name': test.name,
                'success': success,
                'latency': latency,
                'expected_issues': test.expected_issues,
                'issues_found': issues_found,
                'total_issues': num_issues,
                'total_suggestions': num_suggestions,
                'performance': result['performance'],
                'retrieval_count': result['retrieval']['metadata']['retrieval_count'],
                'review_summary': result['review']['summary']
            }
            
            # Log results
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"\n{status}")
            logger.info(f"Expected issues: {test.expected_issues}")
            logger.info(f"Issues found: {issues_found}")
            logger.info(f"Total feedback: {num_issues} issues, {num_suggestions} suggestions")
            logger.info(f"Latency: {latency:.2f}s")
            logger.info(f"Cost: ${result['performance']['cost']:.4f}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            return {
                'name': test.name,
                'success': False,
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    def run_all_tests(self) -> Dict:
        """Run complete test suite."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║         REVIEW QUALITY TEST SUITE                                ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        results = []
        total_cost = 0
        total_time = 0
        
        for test in self.test_cases:
            result = self.run_test_case(test)
            results.append(result)
            
            if 'performance' in result:
                total_cost += result['performance']['cost']
                total_time += result['latency']
            
            # Pause between tests
            time.sleep(1)
        
        # Summary statistics
        self.print_summary(results, total_cost, total_time)
        
        return {
            'results': results,
            'total_cost': total_cost,
            'total_time': total_time,
            'pass_rate': sum(r['success'] for r in results) / len(results)
        }
    
    def print_summary(self, results: List[Dict], total_cost: float, total_time: float):
        """Print test summary."""
        print("\n\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"\n✅ Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"💰 Total cost: ${total_cost:.4f}")
        print(f"⚡ Avg latency: {total_time/total:.2f}s per review")
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        print("-"*70)
        
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"\n{status} {r['name']}")
            
            if 'error' in r:
                print(f"   Error: {r['error']}")
            else:
                print(f"   Expected: {r.get('expected_issues', [])}")
                print(f"   Found: {r.get('issues_found', [])}")
                print(f"   Feedback: {r.get('total_issues', 0)} issues, {r.get('total_suggestions', 0)} suggestions")
                print(f"   Time: {r['latency']:.2f}s")
        
        print("\n" + "="*70)


class PerformanceBenchmark:
    """Benchmark performance metrics."""
    
    def __init__(self):
        self.assistant = CodeReviewAssistant(save_history=False)
    
    def benchmark_retrieval_speed(self, num_queries: int = 20) -> Dict:
        """Benchmark retrieval performance."""
        print("\n🔍 RETRIEVAL SPEED BENCHMARK")
        print("-"*70)
        
        sample_code = "def process_data(items):\n    return [x * 2 for x in items]"
        
        times = []
        for i in range(num_queries):
            start = time.time()
            self.assistant.rag.retrieve_similar_reviews(sample_code)
            times.append(time.time() - start)
            
            if (i + 1) % 5 == 0:
                print(f"Progress: {i+1}/{num_queries}")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n✅ Average retrieval time: {avg_time*1000:.2f}ms")
        print(f"   Min: {min_time*1000:.2f}ms")
        print(f"   Max: {max_time*1000:.2f}ms")
        
        return {
            'avg_ms': avg_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000
        }
    
    def benchmark_end_to_end(self, num_reviews: int = 5) -> Dict:
        """Benchmark complete review pipeline."""
        print("\n⚡ END-TO-END BENCHMARK")
        print("-"*70)
        
        sample_codes = [
            "def add(a, b): return a + b",
            "async def fetch(): return await client.get(url)",
            "class User:\n    def __init__(self, name):\n        self.name = name",
            "data = [x for x in range(100) if x % 2 == 0]",
            "with open('file.txt') as f:\n    content = f.read()"
        ]
        
        results = {
            'retrieval_times': [],
            'generation_times': [],
            'total_times': [],
            'costs': [],
            'token_counts': []
        }
        
        for i, code in enumerate(sample_codes[:num_reviews], 1):
            print(f"\nReview {i}/{num_reviews}")
            
            start = time.time()
            result = self.assistant.review_code(code)
            total_time = time.time() - start
            
            results['total_times'].append(total_time)
            results['costs'].append(result['performance']['cost'])
            results['token_counts'].append(result['performance']['total_tokens'])
        
        avg_time = sum(results['total_times']) / len(results['total_times'])
        avg_cost = sum(results['costs']) / len(results['costs'])
        avg_tokens = sum(results['token_counts']) / len(results['token_counts'])
        
        print(f"\n✅ Average review time: {avg_time:.2f}s")
        print(f"   Average cost: ${avg_cost:.4f}")
        print(f"   Average tokens: {avg_tokens:.0f}")
        
        return {
            'avg_time': avg_time,
            'avg_cost': avg_cost,
            'avg_tokens': avg_tokens,
            'details': results
        }


def run_edge_case_tests():
    """Test edge cases."""
    print("\n🧪 EDGE CASE TESTS")
    print("="*70)
    
    assistant = CodeReviewAssistant(save_history=False)
    
    edge_cases = {
        'empty_code': "",
        'very_short': "x = 1",
        'very_long': "def func():\n" + "    pass\n" * 100,
        'non_python': "SELECT * FROM users;",
        'syntax_error': "def broken(\n    return x",
    }
    
    results = {}
    
    for name, code in edge_cases.items():
        print(f"\n📝 Testing: {name}")
        try:
            result = assistant.review_code(code)
            results[name] = {
                'success': True,
                'has_feedback': len(result['review']['issues']) > 0
            }
            print(f"   ✅ Handled successfully")
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ⚠️  Exception: {e}")
    
    return results


def main():
    """Main test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test review quality')
    parser.add_argument('--quick', action='store_true', help='Run quick test (3 cases)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--edge-cases', action='store_true', help='Test edge cases')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         CODE INTELLIGENCE ASSISTANT - TEST SUITE                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Quality tests
    if not args.benchmark and not args.edge_cases:
        tester = ReviewQualityTester()
        
        if args.quick:
            # Run first 3 tests only
            tester.test_cases = tester.test_cases[:3]
            print("\n🚀 Running QUICK test suite (3 cases)")
        
        results = tester.run_all_tests()
        
        # Save results
        output_file = Path('data/test_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    # Performance benchmarks
    if args.benchmark:
        benchmark = PerformanceBenchmark()
        
        retrieval_perf = benchmark.benchmark_retrieval_speed(num_queries=20)
        e2e_perf = benchmark.benchmark_end_to_end(num_reviews=5)
        
        print("\n📊 BENCHMARK SUMMARY")
        print("="*70)
        print(f"Retrieval: {retrieval_perf['avg_ms']:.2f}ms avg")
        print(f"End-to-end: {e2e_perf['avg_time']:.2f}s avg")
        print(f"Cost per review: ${e2e_perf['avg_cost']:.4f}")
    
    # Edge cases
    if args.edge_cases:
        edge_results = run_edge_case_tests()
        
        print("\n📊 EDGE CASE SUMMARY")
        print("="*70)
        passed = sum(1 for r in edge_results.values() if r['success'])
        print(f"Handled: {passed}/{len(edge_results)}")
    
    print("\n✨ Testing complete!")


if __name__ == '__main__':
    main()