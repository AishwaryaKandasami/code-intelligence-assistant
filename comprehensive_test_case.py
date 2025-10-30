"""
Comprehensive Test Cases Generator
Creates 15 diverse test cases covering all major code review categories.

Usage:
    python comprehensive_test_cases.py
"""

from pathlib import Path
import json

# Create test_cases directory
test_dir = Path('test_cases')
test_dir.mkdir(exist_ok=True)

# Comprehensive test cases with metadata
test_cases = {
    # ========================================================================
    # SECURITY ISSUES (Critical)
    # ========================================================================
    
    'security_01_sql_injection.py': {
        'code': '''def get_user_by_id(user_id):
    """Fetch user from database by ID."""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result.fetchone()
''',
        'category': 'SECURITY',
        'severity': 'CRITICAL',
        'expected_detection': ['SQL injection', 'parameterized query', 'security vulnerability'],
        'description': 'Classic SQL injection vulnerability using string formatting'
    },
    
    'security_02_hardcoded_secrets.py': {
        'code': '''import requests

def connect_to_api():
    """Connect to payment API."""
    api_key = "sk_live_51HxYZ123456789"
    secret = "whsec_abcdef123456"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'X-Secret': secret
    }
    return requests.get('https://api.payment.com/charge', headers=headers)
''',
        'category': 'SECURITY',
        'severity': 'CRITICAL',
        'expected_detection': ['hardcoded credentials', 'secrets', 'environment variables'],
        'description': 'Hardcoded API keys and secrets'
    },
    
    'security_03_path_traversal.py': {
        'code': '''def read_user_file(filename):
    """Read user uploaded file."""
    file_path = f"/uploads/{filename}"
    with open(file_path, 'r') as f:
        return f.read()
''',
        'category': 'SECURITY',
        'severity': 'HIGH',
        'expected_detection': ['path traversal', 'input validation', 'security'],
        'description': 'Path traversal vulnerability - user can access any file'
    },
    
    # ========================================================================
    # BUG / ERROR HANDLING (High Priority)
    # ========================================================================
    
    'bug_01_missing_error_handling.py': {
        'code': '''async def fetch_user_profile(user_id):
    """Fetch user profile from API."""
    response = await http_client.get(f'/users/{user_id}')
    data = response.json()
    return {
        'name': data['profile']['name'],
        'email': data['contact']['email'],
        'avatar': data['profile']['images']['avatar']
    }
''',
        'category': 'BUG',
        'severity': 'HIGH',
        'expected_detection': ['error handling', 'KeyError', 'exception', 'try-except'],
        'description': 'No error handling for nested dict access and network failures'
    },
    
    'bug_02_race_condition.py': {
        'code': '''class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        """Increment counter."""
        current = self.count
        time.sleep(0.001)  # Simulate processing
        self.count = current + 1
        return self.count
''',
        'category': 'BUG',
        'severity': 'HIGH',
        'expected_detection': ['race condition', 'thread-safe', 'lock', 'concurrency'],
        'description': 'Classic race condition in multi-threaded environment'
    },
    
    'bug_03_mutable_default.py': {
        'code': '''def add_to_cart(item, cart=[]):
    """Add item to shopping cart."""
    cart.append(item)
    return cart

def process_order(items, metadata={}):
    """Process order with metadata."""
    metadata['processed'] = True
    metadata['items'] = items
    return metadata
''',
        'category': 'BUG',
        'severity': 'MEDIUM',
        'expected_detection': ['mutable default', 'default argument', 'shared state'],
        'description': 'Dangerous mutable default arguments'
    },
    
    'bug_04_resource_leak.py': {
        'code': '''def process_log_file(filepath):
    """Process application log file."""
    file = open(filepath, 'r')
    lines = []
    
    for line in file:
        if 'ERROR' in line:
            lines.append(line.strip())
    
    if len(lines) > 1000:
        raise ValueError("Too many errors")
    
    return lines
''',
        'category': 'BUG',
        'severity': 'MEDIUM',
        'expected_detection': ['resource leak', 'context manager', 'file not closed', 'with statement'],
        'description': 'File handle not closed, especially on exception'
    },
    
    # ========================================================================
    # PERFORMANCE ISSUES (Medium Priority)
    # ========================================================================
    
    'performance_01_n_squared.py': {
        'code': '''def find_common_elements(list1, list2):
    """Find common elements between two lists."""
    common = []
    for item in list1:
        if item in list2:
            common.append(item)
    return common
''',
        'category': 'PERFORMANCE',
        'severity': 'MEDIUM',
        'expected_detection': ['O(n²)', 'performance', 'set', 'optimization'],
        'description': 'O(n²) algorithm when O(n) is possible with sets'
    },
    
    'performance_02_repeated_computation.py': {
        'code': '''def calculate_prices(items):
    """Calculate prices with tax."""
    results = []
    for item in items:
        base_price = item['price']
        tax = base_price * get_tax_rate()  # DB call each time
        shipping = calculate_shipping(item)  # API call each time
        total = base_price + tax + shipping
        results.append(total)
    return results
''',
        'category': 'PERFORMANCE',
        'severity': 'MEDIUM',
        'expected_detection': ['repeated computation', 'cache', 'optimization'],
        'description': 'Expensive operations repeated in loop'
    },
    
    # ========================================================================
    # STYLE / MAINTAINABILITY (Low Priority)
    # ========================================================================
    
    'style_01_missing_type_hints.py': {
        'code': '''def calculate_discount(price, customer_type, quantity, coupon_code):
    """Calculate final price with discounts."""
    base_discount = 0.05 if customer_type == 'regular' else 0.15
    
    if quantity > 100:
        base_discount += 0.10
    elif quantity > 50:
        base_discount += 0.05
    
    if coupon_code:
        base_discount += 0.05
    
    return price * (1 - base_discount)
''',
        'category': 'STYLE',
        'severity': 'LOW',
        'expected_detection': ['type hints', 'type annotation', 'int', 'float', 'str'],
        'description': 'Missing type hints for parameters and return value'
    },
    
    'style_02_complex_function.py': {
        'code': '''def process_payment(order_id, user_id, amount, payment_method, billing_address, shipping_address, coupon_code, gift_wrap, express_shipping, gift_message, newsletter_signup):
    """Process payment for order."""
    user = db.get_user(user_id)
    if user.status != 'active':
        if user.status == 'suspended':
            return {'error': 'Account suspended'}
        elif user.status == 'pending':
            return {'error': 'Account pending verification'}
        else:
            return {'error': 'Invalid account'}
    
    if payment_method == 'credit_card':
        if not validate_credit_card(user.card):
            return {'error': 'Invalid card'}
    elif payment_method == 'paypal':
        if not user.paypal_linked:
            return {'error': 'PayPal not linked'}
    
    total = amount
    if coupon_code:
        discount = get_coupon_discount(coupon_code)
        if discount:
            total = total * (1 - discount)
    
    if gift_wrap:
        total += 5
    if express_shipping:
        total += 15
    
    return {'success': True, 'total': total}
''',
        'category': 'MAINTAINABILITY',
        'severity': 'MEDIUM',
        'expected_detection': ['complex', 'refactor', 'too many parameters', 'nested conditions'],
        'description': 'Overly complex function with too many parameters and nested logic'
    },
    
    # ========================================================================
    # DOCUMENTATION (Low Priority)
    # ========================================================================
    
    'documentation_01_missing_docstring.py': {
        'code': '''def validate_email(email):
    if not email or '@' not in email:
        return False
    
    parts = email.split('@')
    if len(parts) != 2:
        return False
    
    local, domain = parts
    if not local or not domain or '.' not in domain:
        return False
    
    return True
''',
        'category': 'DOCUMENTATION',
        'severity': 'LOW',
        'expected_detection': ['docstring', 'documentation', 'comment', 'explain'],
        'description': 'No docstring explaining validation rules'
    },
    
    # ========================================================================
    # GOOD CODE (Baseline / Control)
    # ========================================================================
    
    'good_code_01_well_written.py': {
        'code': '''from typing import List, Optional
from decimal import Decimal

def calculate_total_price(
    items: List[dict],
    tax_rate: Decimal,
    discount: Optional[Decimal] = None
) -> Decimal:
    """
    Calculate total price with tax and optional discount.
    
    Args:
        items: List of items with 'price' and 'quantity' keys
        tax_rate: Tax rate as decimal (e.g., 0.08 for 8%)
        discount: Optional discount rate (e.g., 0.10 for 10% off)
        
    Returns:
        Total price including tax and discount
        
    Raises:
        ValueError: If items list is empty or contains invalid data
    """
    if not items:
        raise ValueError("Items list cannot be empty")
    
    subtotal = sum(
        Decimal(str(item['price'])) * item['quantity']
        for item in items
    )
    
    if discount:
        subtotal *= (Decimal('1') - discount)
    
    total = subtotal * (Decimal('1') + tax_rate)
    return total.quantize(Decimal('0.01'))
''',
        'category': 'GOOD_CODE',
        'severity': 'NONE',
        'expected_detection': [],
        'description': 'Well-written code with type hints, docstring, error handling'
    },
    
    'good_code_02_modern_python.py': {
        'code': '''from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    """User model with validation."""
    id: int
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    
    def __post_init__(self):
        """Validate user data."""
        if not self.email or '@' not in self.email:
            raise ValueError(f"Invalid email: {self.email}")
        
        if not self.name or len(self.name) < 2:
            raise ValueError("Name must be at least 2 characters")
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
    
    @property
    def age_days(self) -> int:
        """Get account age in days."""
        return (datetime.now() - self.created_at).days
''',
        'category': 'GOOD_CODE',
        'severity': 'NONE',
        'expected_detection': [],
        'description': 'Modern Python with dataclasses, type hints, validation'
    }
}

# Save test cases
print("📝 Creating Comprehensive Test Cases...")
print("=" * 70)

metadata = {
    'created_at': '2025-10-27',
    'total_tests': len(test_cases),
    'categories': {},
    'tests': []
}

for filename, test_data in test_cases.items():
    # Save code file
    filepath = test_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(test_data['code'].strip() + '\n')
    
    # Track metadata
    category = test_data['category']
    if category not in metadata['categories']:
        metadata['categories'][category] = 0
    metadata['categories'][category] += 1
    
    metadata['tests'].append({
        'filename': filename,
        'category': category,
        'severity': test_data['severity'],
        'description': test_data['description'],
        'expected_detection': test_data['expected_detection']
    })
    
    # Print progress
    lines = len(test_data['code'].strip().split('\n'))
    severity_emoji = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🟢',
        'NONE': '⚪'
    }
    emoji = severity_emoji.get(test_data['severity'], '⚪')
    
    print(f"{emoji} {filename:<45} ({lines:2} lines) [{category}]")

# Save metadata
metadata_file = test_dir / 'test_metadata.json'
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 70)
print(f"✅ Created {len(test_cases)} test files in '{test_dir}/' directory")
print("\n📊 Test Distribution:")
print("-" * 70)

for category, count in sorted(metadata['categories'].items()):
    print(f"   {category:<20} : {count:2} tests")

print("\n" + "=" * 70)
print("\n🚀 How to Use:")
print("=" * 70)
print("""
1️⃣  Test Individual File:
   python scripts/07_endreview.py --file test_cases/security_01_sql_injection.py

2️⃣  Interactive Mode:
   python scripts/07_endreview.py --interactive
   Then: file test_cases/security_02_hardcoded_secrets.py

3️⃣  Batch Test All:
   python batch_test_all.py

4️⃣  Direct Code Test:
   python scripts/07_endreview.py --code "def add(a, b): return a + b"
""")

print("=" * 70)
print("✨ Ready for comprehensive testing!")
print("=" * 70)