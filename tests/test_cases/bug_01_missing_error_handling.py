async def fetch_user_profile(user_id):
    """Fetch user profile from API."""
    response = await http_client.get(f'/users/{user_id}')
    data = response.json()
    return {
        'name': data['profile']['name'],
        'email': data['contact']['email'],
        'avatar': data['profile']['images']['avatar']
    }
