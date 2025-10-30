def process_log_file(filepath):
    """Process application log file."""
    file = open(filepath, 'r')
    lines = []
    
    for line in file:
        if 'ERROR' in line:
            lines.append(line.strip())
    
    if len(lines) > 1000:
        raise ValueError("Too many errors")
    
    return lines
