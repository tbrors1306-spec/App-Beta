import unittest
import pandas as pd
import math
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.calculations import PipeCalculator

class TestPipeCalculator(unittest.TestCase):
    def setUp(self):
        # Create a mock dataframe similar to the real one
        data = {
            'DN': [100],
            'D_Aussen': [114.3],
            'Radius_BA3': [152],
            'T_Stueck_H': [105],
            'Red_Laenge_L': [102],
            'Flansch_b_16': [52],
            'Flansch_b_10': [52] # For PN 10
        }
        self.df = pd.DataFrame(data)
        self.calc = PipeCalculator(self.df)

    def test_get_deduction_90_deg_bend(self):
        # Bogen 90° (BA3) should return Radius_BA3
        deduction = self.calc.get_deduction("Bogen 90° (BA3)", 100, "PN 16")
        self.assertEqual(deduction, 152.0)

    def test_get_deduction_custom_angle(self):
        # Bogen (Zuschnitt) 45°
        # Formula: Radius_BA3 * tan(angle/2)
        # 152 * tan(22.5) = 152 * 0.4142... approx 62.9
        deduction = self.calc.get_deduction("Bogen (Zuschnitt)", 100, "PN 16", 45.0)
        expected = 152 * math.tan(math.radians(22.5))
        self.assertAlmostEqual(deduction, expected, places=1)

    def test_calculate_rolling_offset_simple(self):
        # 3, 4, 5 triangle
        # roll=400, set=300 -> diag=500
        # height=0 -> travel=500
        res = self.calc.calculate_rolling_offset(100, 400, 300)
        self.assertEqual(res['diag_base'], 500.0)
        self.assertEqual(res['travel'], 500.0)
        self.assertAlmostEqual(res['angle_calc'], 36.87, places=2)

    def test_calculate_rolling_offset_impossible(self):
        # Should handle division by zero or invalid acos domain gracefully
        # Case: travel=0
        res = self.calc.calculate_rolling_offset(100, 0, 0, 0)
        self.assertEqual(res['travel'], 0)
        self.assertEqual(res['angle_calc'], 0)

    def test_calculate_2d_offset_zero(self):
        # Zero angle should return error dict
        res = self.calc.calculate_2d_offset(100, 500, 0)
        self.assertIn("error", res)

if __name__ == '__main__':
    unittest.main()
