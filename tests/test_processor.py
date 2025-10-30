"""Test cases for data processor."""

def test_comment_quality():
    """Test comment filtering logic."""
    
    # Should pass
    good_comments = [
        "Consider using async here for better performance",
        "This could be simplified by using a list comprehension",
        "Great catch! But we should also handle the None case"
    ]
    
    # Should fail
    bad_comments = [
        "lgtm",
        "ok",
        "👍",
        "a",  # too short
        "thanks"
    ]
    
    print("✅ Good comments:", len(good_comments))
    print("❌ Bad comments:", len(bad_comments))
    print("Test cases ready!")

if __name__ == '__main__':
    test_comment_quality()