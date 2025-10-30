def read_user_file(filename):
    """Read user uploaded file."""
    file_path = f"/uploads/{filename}"
    with open(file_path, 'r') as f:
        return f.read()
