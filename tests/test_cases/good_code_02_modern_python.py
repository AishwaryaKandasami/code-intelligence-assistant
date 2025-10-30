from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    """User model with validation."""
    id: int
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    
    def __post_init__(self):
        """Validate user data."""
        if not self.email or '@' not in self.email:
            raise ValueError(f"Invalid email: {self.email}")
        
        if not self.name or len(self.name) < 2:
            raise ValueError("Name must be at least 2 characters")
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
    
    @property
    def age_days(self) -> int:
        """Get account age in days."""
        return (datetime.now() - self.created_at).days
