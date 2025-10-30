def fetch_user_profile(user_id):
    import requests
    
    # 🔹 Issue 1: Magic string (style issue)
    response = requests.get(f"https://api.example.com/users/{user_id}")
    
    # 🔹 Issue 2: Potential security / KeyError
    data = response.json()
    email = data['profile']['contact']['email']  # unsafe access, could raise KeyError
    
    return email

def main():
    user_id = 123
    print(fetch_user_profile(user_id))

if __name__ == "__main__":
    main()
