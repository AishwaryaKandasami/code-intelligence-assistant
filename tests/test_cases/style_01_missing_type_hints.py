def calculate_discount(price, customer_type, quantity, coupon_code):
    """Calculate final price with discounts."""
    base_discount = 0.05 if customer_type == 'regular' else 0.15
    
    if quantity > 100:
        base_discount += 0.10
    elif quantity > 50:
        base_discount += 0.05
    
    if coupon_code:
        base_discount += 0.05
    
    return price * (1 - base_discount)
