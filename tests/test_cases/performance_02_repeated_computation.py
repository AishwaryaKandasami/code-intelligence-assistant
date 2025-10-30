def calculate_prices(items):
    """Calculate prices with tax."""
    results = []
    for item in items:
        base_price = item['price']
        tax = base_price * get_tax_rate()  # DB call each time
        shipping = calculate_shipping(item)  # API call each time
        total = base_price + tax + shipping
        results.append(total)
    return results
