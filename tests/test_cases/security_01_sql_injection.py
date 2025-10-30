def get_user_by_id(user_id):
    """Fetch user from database by ID."""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result.fetchone()
