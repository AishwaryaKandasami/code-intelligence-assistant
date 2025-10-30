def find_common_elements(list1, list2):
    """Find common elements between two lists."""
    common = []
    for item in list1:
        if item in list2:
            common.append(item)
    return common
