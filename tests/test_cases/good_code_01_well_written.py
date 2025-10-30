from typing import List, Optional
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
