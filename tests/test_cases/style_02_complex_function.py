def process_payment(order_id, user_id, amount, payment_method, billing_address, shipping_address, coupon_code, gift_wrap, express_shipping, gift_message, newsletter_signup):
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
