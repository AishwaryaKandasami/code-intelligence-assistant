class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        """Increment counter."""
        current = self.count
        time.sleep(0.001)  # Simulate processing
        self.count = current + 1
        return self.count
