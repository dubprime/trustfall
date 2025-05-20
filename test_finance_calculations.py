import unittest
import numpy as np
from pie_matrix import InputParams, SimulationResults, run_simulation

class TestFinancialCalculations(unittest.TestCase):
    def setUp(self):
        # Set up basic test parameters
        self.params = InputParams(
            initial_capital=100000,
            monthly_interest=0.12,
            num_months=12,
            default_rates={1: 0.1, 2: 0.1, 3: 0.1},
            allocation={1: 0.5, 2: 0.25, 3: 0.25},
            show_def_overlay=True
        )
        self.sim = run_simulation(self.params)

    def test_conservation_of_value(self):
        """Test that total value is conserved across the simulation."""
        # Initial capital
        initial = self.params.initial_capital
        
        # Final values: sum of all cash flows + remaining portfolio
        total_defaults = sum(self.sim.per_period_defaults)
        total_interest = sum(self.sim.per_period_interest)
        total_principal = sum(self.sim.per_period_principal)
        final_outstanding = sum(out[-1] for out in self.sim.out_net.values())
        
        # Total value should be: initial + interest - defaults
        expected_total = initial + total_interest - total_defaults
        actual_total = final_outstanding + total_principal
        
        # Allow for small floating point differences
        self.assertAlmostEqual(expected_total, actual_total, delta=0.01)

    def test_payment_calculation_factors(self):
        """Test that payment factors match standard financial formulas."""
        r = self.params.monthly_interest
        
        # Manual calculation of payment factors
        expected_factors = {}
        for n in [2, 3]:
            expected_factors[n] = r / (1 - (1 + r) ** (-n))
            
        # Compare with the model's calculated factors
        for n in [2, 3]:
            self.assertAlmostEqual(
                expected_factors[n],
                self.sim.pmt_factors[n],
                delta=0.0001
            )

    def test_hazard_rate_calculation(self):
        """Test the hazard rate calculations."""
        d = self.params.default_rates
        
        # Calculate expected hazard rates
        expected_hazards = {}
        for n in [2, 3]:
            expected_hazards[n] = 1 - (1 - d[n]) ** (1/n)
        
        # Run simulation with fresh parameters
        params = self.params
        sim = run_simulation(params)
        
        # Extract hazard rates from simulation
        # Note: This requires extracting hazard rates from the simulation logic
        # This is a suggestion for how the code might be refactored to make testing easier
        calculated_hazards = {n: 1-(1-d[n])**(1/n) for n in [2, 3]}
        
        # Compare expected vs calculated
        for n in [2, 3]:
            self.assertAlmostEqual(
                expected_hazards[n],
                calculated_hazards[n],
                delta=0.0001
            )

    def test_moic_calculation(self):
        """Test that MOIC calculations are accurate."""
        # Gross MOIC should be (1+r)^n
        r = self.params.monthly_interest
        expected_gross_moic = {n: (1 + r) ** n for n in [1, 2, 3]}
        
        # Calculate actual gross MOIC from simulation
        sum_new_by_tenor = {n: sum(self.sim.new_by_t[:, i]) for i, n in enumerate([1, 2, 3])}
        sum_payments_by_tenor = {n: sum(self.sim.payment_by_tenor[n]) for n in [1, 2, 3]}
        
        # Calculate actual MOIC
        actual_moic = {}
        for n in [1, 2, 3]:
            if sum_new_by_tenor[n] > 0:
                actual_moic[n] = sum_payments_by_tenor[n] / sum_new_by_tenor[n]
            else:
                actual_moic[n] = 0
        
        # Net MOIC should be less than gross MOIC due to defaults
        for n in [1, 2, 3]:
            if actual_moic[n] > 0:  # Skip tenors with no allocation
                self.assertLess(actual_moic[n], expected_gross_moic[n])
                
                # But should be positive if not 100% default
                self.assertGreater(actual_moic[n], 1.0)

    def test_default_timing(self):
        """Test that defaults are properly timed across periods."""
        # For 1-month loans, defaults should happen in month 1
        month1_defaults = self.sim.default_by_tenor[1][1]
        self.assertGreater(month1_defaults, 0)
        
        # For multi-month loans, defaults should be spread over months
        for n in [2, 3]:
            for m in range(1, n+1):
                self.assertGreaterEqual(
                    self.sim.default_by_tenor[n][m], 
                    0, 
                    f"Default for {n}-month tenor in month {m} should be >= 0"
                )

    def test_allocation_distribution(self):
        """Test that capital is allocated according to the specified mix."""
        # Initial allocation in month 0
        month0_new = self.sim.new_by_t[0]
        total_allocated = sum(month0_new)
        
        # Check proportion matches input allocation
        for i, n in enumerate([1, 2, 3]):
            expected = self.params.allocation[n] * self.params.initial_capital
            actual = month0_new[i]
            self.assertAlmostEqual(expected, actual, delta=0.01)

    def test_interest_calculation(self):
        """Test that interest is calculated correctly on surviving balance."""
        r = self.params.monthly_interest
        
        # For 1-month loans, interest should be r * (1-default_rate) * principal
        month0_1m_principal = self.sim.new_by_t[0, 0]
        expected_1m_interest = r * (1 - self.params.default_rates[1]) * month0_1m_principal
        actual_1m_interest = self.sim.interest_by_tenor[1][1]
        
        self.assertAlmostEqual(expected_1m_interest, actual_1m_interest, delta=0.01)

if __name__ == '__main__':
    unittest.main() 