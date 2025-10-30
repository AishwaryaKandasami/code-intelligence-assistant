def add_to_cart(item, cart=[]):
    """Add item to shopping cart."""
    cart.append(item)
    return cart

def process_order(items, metadata={}):
    """Process order with metadata."""
    metadata['processed'] = True
    metadata['items'] = items
    return metadata
