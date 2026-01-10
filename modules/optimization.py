from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class CutRequest:
    id: str  # e.g., "Schnitt A" or just ID
    length: float

@dataclass
class OptBar:
    id: int
    length: float
    cuts: List[CutRequest]
    waste: float

class CuttingOptimizer:
    @staticmethod
    def solve_ffd(cut_requests: List[CutRequest], stock_length: float, saw_width: float = 3.0) -> List[OptBar]:
        """
        Solves the Bin Packing problem using First Fit Decreasing (FFD).
        """
        # 1. Sort cuts descending
        sorted_cuts = sorted(cut_requests, key=lambda x: x.length, reverse=True)
        
        bars: List[OptBar] = []
        
        for cut in sorted_cuts:
            placed = False
            needed = cut.length
            
            # 2. Try to fit into existing bars
            for bar in bars:
                # Actual space used = cut length + saw width (if not the first cut, but simplified: always add width for every cut except last? 
                # Better: remaining space calculation.
                # Space calculation: current_used = sum(c.length for c in bar.cuts) + (len(bar.cuts) * saw_width)
                # Available?
                
                current_used = sum(c.length for c in bar.cuts) + (len(bar.cuts) * saw_width)
                space_left = stock_length - current_used
                
                # Check if fits (including saw width if it's not the very first item in empty bar, assuming cuts are sequential)
                # Actually, effectively each cut 'consumes' length + saw_width, except maybe the last piece doesn't need a cut after it?
                # Conservative approach: Assume every cut consumes length + saw_width.
                if space_left >= (needed + saw_width):
                    bar.cuts.append(cut)
                    # Recalculate waste
                    new_used = current_used + needed + saw_width
                    bar.waste = stock_length - new_used
                    placed = True
                    break
                elif len(bar.cuts) == 0 and space_left >= needed: # Empty bar, fits exactly or without saw width?
                     # Technically first cut doesn't strictly need saw width 'before' it if we trim, but usually we ignore start-trim.
                     # Let's simple apply saw_width to every cut involved to be safe (blade thickness).
                     if space_left >= (needed + saw_width):
                        bar.cuts.append(cut)
                        bar.waste = stock_length - (needed + saw_width)
                        placed = True
                        break

            # 3. If not placed, start new bar
            if not placed:
                new_bar = OptBar(id=len(bars)+1, length=stock_length, cuts=[cut], waste=0)
                used = needed + saw_width
                new_bar.waste = stock_length - used
                bars.append(new_bar)
                
        return bars
