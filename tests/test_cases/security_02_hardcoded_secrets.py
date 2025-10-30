import requests

def connect_to_api():
    """Connect to payment API."""
    api_key = "sk_live_51HxYZ123456789"
    secret = "whsec_abcdef123456"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'X-Secret': secret
    }
    return requests.get('https://api.payment.com/charge', headers=headers)
