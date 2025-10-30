"""
Test Cases Suite Generator
Creates 10 test files with various code issues for testing the review system.

Usage:
    python create_test_cases.py
"""

from pathlib import Path

# Create test_cases directory
test_dir = Path('test_cases')
test_dir.mkdir(exist_ok=True)

# =============================================================================
# TEST CASE 1: SQL Injection (5 lines) - SECURITY
# =============================================================================

test1 = '''def get_user_by_id(user_id):
    """Fetch user from database by ID."""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result.fetchone()
'''

# =============================================================================
# TEST CASE 2: Missing Error Handling (7 lines) - BUG
# =============================================================================

test2 = '''async def fetch_api_data(url):
    """Fetch data from external API."""
    response = await http_client.get(url)
    data = response.json()
    user_id = data['user']['id']
    profile = data['user']['profile']['avatar']
    return {'id': user_id, 'avatar': profile}
'''

# =============================================================================
# TEST CASE 3: Performance Issue - O(n²) (8 lines) - PERFORMANCE
# =============================================================================

test3 = '''def remove_duplicates(items):
    """Remove duplicate items from list."""
    unique = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique
'''

# =============================================================================
# TEST CASE 4: Missing Type Hints (6 lines) - STYLE
# =============================================================================

test4 = '''def calculate_discount(price, customer_type, quantity):
    """Calculate discount based on customer type and quantity."""
    discount = 0.05 if customer_type == 'regular' else 0.15
    if quantity > 100:
        discount += 0.10
    return price * (1 - discount)
'''

# =============================================================================
# TEST CASE 5: Hardcoded Credentials (9 lines) - SECURITY
# =============================================================================

test5 = '''import requests

def connect_to_database():
    """Establish database connection."""
    host = "prod-db.company.com"
    username = "admin"
    password = "Admin123!"
    connection_string = f"postgresql://{username}:{password}@{host}/maindb"
    return create_connection(connection_string)
'''

# =============================================================================
# TEST CASE 6: Race Condition (10 lines) - BUG
# =============================================================================

test6 = '''class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        """Withdraw money from account."""
        if self.balance >= amount:
            time.sleep(0.1)  # Simulate processing
            self.balance -= amount
            return True
        return False
'''

# =============================================================================
# TEST CASE 7: Missing Documentation (12 lines) - DOCUMENTATION
# =============================================================================

test7 = '''def process_payment(order_id, amount, payment_method, customer_id):
    order = get_order(order_id)
    customer = get_customer(customer_id)
    
    if not validate_payment_method(payment_method):
        return {'status': 'error', 'message': 'Invalid payment method'}
    
    if customer.balance < amount:
        return {'status': 'error', 'message': 'Insufficient funds'}
    
    transaction = create_transaction(order_id, amount, payment_method)
    return {'status': 'success', 'transaction_id': transaction.id}
'''

# =============================================================================
# TEST CASE 8: Mutable Default Argument (6 lines) - BUG
# =============================================================================

test8 = '''def add_item_to_cart(item, cart=[]):
    """Add an item to shopping cart."""
    cart.append(item)
    total = sum(i['price'] for i in cart)
    return {'cart': cart, 'total': total}
'''

# =============================================================================
# TEST CASE 9: Complex Nested Logic (15 lines) - MAINTAINABILITY
# =============================================================================

test9 = '''def calculate_shipping_cost(weight, country, shipping_type, is_member):
    """Calculate shipping cost based on multiple factors."""
    if country == 'US':
        if shipping_type == 'express':
            cost = 25 if weight < 5 else 50
        else:
            cost = 10 if weight < 5 else 20
    elif country in ['CA', 'MX']:
        if shipping_type == 'express':
            cost = 35 if weight < 5 else 70
        else:
            cost = 15 if weight < 5 else 30
    else:
        cost = 100
    return cost * 0.9 if is_member else cost
'''

# =============================================================================
# TEST CASE 10: Resource Leak (11 lines) - BUG
# =============================================================================

test10 = '''def process_large_file(filepath):
    """Process a large data file."""
    file = open(filepath, 'r')
    data = []
    for line in file:
        if line.strip():
            processed = line.strip().upper()
            data.append(processed)
    if len(data) > 1000:
        raise ValueError("File too large")
    return data
'''

# =============================================================================
# Save all test cases
# =============================================================================

test_cases = {
    'test1_sql_injection.py': test1,
    'test2_error_handling.py': test2,
    'test3_performance.py': test3,
    'test4_type_hints.py': test4,
    'test5_hardcoded_secrets.py': test5,
    'test6_race_condition.py': test6,
    'test7_missing_docs.py': test7,
    'test8_mutable_default.py': test8,
    'test9_complex_logic.py': test9,
    'test10_resource_leak.py': test10,
}

print("📝 Creating test cases...")
print("=" * 70)

for filename, code in test_cases.items():
    filepath = test_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code.strip() + '\n')
    
    lines = len(code.strip().split('\n'))
    print(f"✅ {filename:<30} ({lines:2} lines)")

print("\n" + "=" * 70)
print(f"✅ Created {len(test_cases)} test files in '{test_dir}/' directory")
print("\n📋 Test Case Summary:")
print("=" * 70)

summaries = [
    ("test1_sql_injection.py", "SQL Injection vulnerability", "SECURITY"),
    ("test2_error_handling.py", "Missing error/null checks", "BUG"),
    ("test3_performance.py", "O(n²) performance issue", "PERFORMANCE"),
    ("test4_type_hints.py", "Missing type annotations", "STYLE"),
    ("test5_hardcoded_secrets.py", "Hardcoded credentials", "SECURITY"),
    ("test6_race_condition.py", "Thread-safety issue", "BUG"),
    ("test7_missing_docs.py", "Missing docstrings/comments", "DOCUMENTATION"),
    ("test8_mutable_default.py", "Mutable default argument", "BUG"),
    ("test9_complex_logic.py", "High cyclomatic complexity", "MAINTAINABILITY"),
    ("test10_resource_leak.py", "File handle not closed", "BUG"),
]

for filename, description, category in summaries:
    print(f"{filename:<30} - {description:<35} [{category}]")

print("\n" + "=" * 70)
print("\n🚀 How to Use These Test Cases:")
print("=" * 70)
print("\n1️⃣  Test Individual File:")
print("   python scripts/07_endreview.py --file test_cases/test1_sql_injection.py")
print("\n2️⃣  Test with Context:")
print("   python scripts/07_endreview.py --file test_cases/test5_hardcoded_secrets.py --context \"Production API code\"")
print("\n3️⃣  Interactive Mode:")
print("   python scripts/07_endreview.py --interactive")
print("   Then: file test_cases/test3_performance.py")
print("\n4️⃣  Save Output:")
print("   python scripts/07_endreview.py --file test_cases/test1_sql_injection.py --output reviews/test1_review.json")
print("\n5️⃣  Streaming Mode:")
print("   python scripts/07_endreview.py --file test_cases/test2_error_handling.py --stream")

print("\n" + "=" * 70)
print("\n📊 Batch Testing Script:")
print("=" * 70)
print("""
# Create batch test script
cat > test_all.sh << 'EOF'
#!/bin/bash
mkdir -p reviews

for file in test_cases/*.py; do
    echo "Testing: $file"
    python scripts/07_endreview.py --file "$file" --output "reviews/$(basename "$file" .py)_review.json"
    echo "---"
done

echo "All tests complete! Check reviews/ directory"
EOF

chmod +x test_all.sh
./test_all.sh
""")

print("\n" + "=" * 70)
print("✨ Test cases ready! Start reviewing! 🚀")
print("=" * 70)