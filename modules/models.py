from dataclasses import dataclass, field
from typing import List

@dataclass
class FittingItem:
    id: str
    name: str
    count: int
    deduction_single: float
    dn: int
    
    @property
    def total_deduction(self) -> float: 
        return self.deduction_single * self.count

@dataclass
class SavedCut:
    id: int
    name: str
    raw_length: float
    cut_length: float
    details: str
    timestamp: str
    fittings: List[FittingItem] = field(default_factory=list)
