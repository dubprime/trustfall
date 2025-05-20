import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pie_matrix import InputParams, SimulationResults, run_simulation
from financial_metrics import compute_extended_metrics, format_metrics_for_display
from stress_testing import (
    StressTestScenario, run_stress_test, run_multiple_stress_tests,
    get_predefined_scenarios, generate_stress_test_report
)

class TestPortfolioSimulatorIntegration(unittest.TestCase):
    """Integration tests for the entire portfolio simulator system."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.base_params = InputParams(
            initial_capital=100000,
            monthly_interest=0.10,  # 10% monthly interest
            num_months=12,
            default_rates={1: 0.05, 2: 0.08, 3: 0.12},  # Increasing default with tenor
            allocation={1: 0.4, 2: 0.3, 3: 0.3},  # Balanced allocation
            show_def_overlay=True
        )
        
        # Run simulation with base parameters
        self.base_sim = run_simulation(self.base_params)
        
        # Compute extended metrics
        self.base_metrics = compute_extended_metrics(self.base_params, self.base_sim)
    
    def test_simulation_results_consistency(self):
        """Test that simulation results are consistent and financially sound."""
        # Check that initial capital is deployed correctly
        month0_allocations = self.base_sim.new_by_t[0]
        total_allocated = sum(month0_allocations)
        self.assertAlmostEqual(total_allocated, self.base_params.initial_capital, delta=0.01)
        
        # Check that tenor allocations follow specified ratios
        for i, tenor in enumerate([1, 2, 3]):
            expected_allocation = self.base_params.allocation[tenor] * self.base_params.initial_capital
            self.assertAlmostEqual(month0_allocations[i], expected_allocation, delta=0.01)
        
        # Conservation of value: total cash flows + ending portfolio = initial + returns - defaults
        total_defaults = sum(self.base_sim.per_period_defaults)
        total_interest = sum(self.base_sim.per_period_interest)
        total_principal = sum(self.base_sim.per_period_principal)
        final_portfolio = sum(value[-1] for value in self.base_sim.out_net.values())
        
        expected_total = self.base_params.initial_capital + total_interest - total_defaults
        actual_total = final_portfolio + total_principal
        
        self.assertAlmostEqual(expected_total, actual_total, delta=1.0)
        
    def test_financial_metrics_calculation(self):
        """Test that extended financial metrics are calculated correctly."""
        # IRR should be positive for a profitable portfolio
        self.assertGreater(self.base_metrics.portfolio_irr, 0)
        
        # IRR should be less than gross interest rate due to defaults
        annualized_gross_rate = (1 + self.base_params.monthly_interest) ** 12 - 1
        annualized_net_irr = (1 + self.base_metrics.portfolio_irr) ** 12 - 1
        self.assertLess(annualized_net_irr, annualized_gross_rate)
        
        # WAL should be within the range of tenors
        self.assertGreaterEqual(self.base_metrics.portfolio_wal, 0)
        self.assertLessEqual(self.base_metrics.portfolio_wal, 3)
        
        # Recovery rates should align with default rates
        for tenor in [1, 2, 3]:
            expected_recovery = 1.0 - self.base_params.default_rates[tenor]
            # Allow for some variation due to cash flow timing
            self.assertAlmostEqual(
                self.base_metrics.recovery_rates[tenor],
                expected_recovery,
                delta=0.2
            )
    
    def test_stress_testing(self):
        """Test that stress testing produces expected impacts."""
        # Define a high default scenario
        high_default_scenario = StressTestScenario(
            name="High Default Test",
            description="Test scenario with doubled default rates",
            default_rate_multiplier=2.0,
            interest_rate_delta=0.0,
            allocation_shift=None
        )
        
        # Run stress test
        stress_result = run_stress_test(self.base_params, high_default_scenario)
        
        # Verify that IRR is lower under stress
        self.assertLess(
            stress_result.stressed_metrics.portfolio_irr,
            stress_result.base_metrics.portfolio_irr
        )
        
        # Verify that default impact is negative (higher defaults = worse outcomes)
        self.assertLess(stress_result.impact_summary["Default Impact (%)"], 0)
        
        # Interest rate hike scenario
        rate_hike_scenario = StressTestScenario(
            name="Rate Hike Test",
            description="Test scenario with increased interest rates",
            default_rate_multiplier=1.0,
            interest_rate_delta=0.02,
            allocation_shift=None
        )
        
        # Run stress test
        rate_result = run_stress_test(self.base_params, rate_hike_scenario)
        
        # Verify that IRR is higher with higher interest rates
        self.assertGreater(
            rate_result.stressed_metrics.portfolio_irr,
            rate_result.base_metrics.portfolio_irr
        )
    
    def test_multiple_stress_scenarios(self):
        """Test running multiple stress scenarios and generating reports."""
        # Get predefined scenarios
        scenarios = get_predefined_scenarios()
        
        # Run multiple stress tests
        results = run_multiple_stress_tests(self.base_params, scenarios[:3])  # Use first 3 scenarios
        
        # Verify results are returned for each scenario
        self.assertEqual(len(results), 3)
        
        # Generate report
        report = generate_stress_test_report(self.base_params, results)
        
        # Verify report has expected structure
        self.assertIsInstance(report, pd.DataFrame)
        self.assertEqual(len(report), 3)
        
        # Check that scenario descriptions match
        for i, scenario in enumerate(scenarios[:3]):
            self.assertEqual(report.iloc[i]["Scenario"], scenario.name)
    
    def test_portfolio_metrics_display(self):
        """Test formatting metrics for display."""
        # Format metrics for display
        display_metrics = format_metrics_for_display(self.base_metrics)
        
        # Verify expected keys are present
        expected_keys = [
            "IRR (%)", "Portfolio IRR", "Sharpe Ratios", 
            "Recovery Rates (%)", "Weighted Average Life (months)",
            "Portfolio WAL", "Concentration Risk", "Cash Flow Volatility"
        ]
        
        for key in expected_keys:
            self.assertIn(key, display_metrics)
        
        # Verify tenor-specific metrics have values for each tenor
        for tenor_dict_key in ["IRR (%)", "Sharpe Ratios", "Recovery Rates (%)"]:
            tenor_dict = display_metrics[tenor_dict_key]
            for tenor in [1, 2, 3]:
                self.assertIn(f"{tenor}-Month", tenor_dict)

    def test_end_to_end_workflow(self):
        """Test the entire workflow from simulation to metrics to stress testing."""
        # Set up test parameters similar to what a user might input
        test_params = InputParams(
            initial_capital=250000,
            monthly_interest=0.12,  # 12% monthly interest
            num_months=24,
            default_rates={1: 0.10, 2: 0.15, 3: 0.20},  # High risk portfolio
            allocation={1: 0.6, 2: 0.3, 3: 0.1},  # Short-term focused
            show_def_overlay=True
        )
        
        # Run simulation
        sim = run_simulation(test_params)
        
        # Check basic simulation results
        self.assertEqual(len(sim.months), test_params.num_months)
        self.assertEqual(sim.new_by_t.shape, (test_params.num_months, 3))
        
        # Calculate extended metrics
        metrics = compute_extended_metrics(test_params, sim)
        
        # Format metrics for display
        display_metrics = format_metrics_for_display(metrics)
        
        # Verify portfolio IRR is formatted as a percentage string
        self.assertIsInstance(display_metrics["Portfolio IRR"], str)
        self.assertIn("%", display_metrics["Portfolio IRR"])
        
        # Run stress tests with custom scenarios
        scenarios = [
            StressTestScenario(
                name="Mild Recession",
                description="Mild economic downturn",
                default_rate_multiplier=1.5,
                interest_rate_delta=-0.01,
                allocation_shift={1: 0.1, 3: -0.1}
            ),
            StressTestScenario(
                name="Severe Recession",
                description="Severe economic downturn",
                default_rate_multiplier=3.0,
                interest_rate_delta=-0.03,
                allocation_shift={1: 0.2, 3: -0.2}
            )
        ]
        
        # Run multiple stress tests
        stress_results = run_multiple_stress_tests(test_params, scenarios)
        
        # Generate report
        report = generate_stress_test_report(test_params, stress_results)
        
        # Verify report has the expected structure
        self.assertEqual(len(report), len(scenarios))
        self.assertTrue(all(scenario.name in report["Scenario"].values for scenario in scenarios))
        
        # Verify severe recession has more negative impact on IRR than mild recession
        mild_change = float(report[report["Scenario"] == "Mild Recession"]["IRR Change"].values[0].strip('%'))
        severe_change = float(report[report["Scenario"] == "Severe Recession"]["IRR Change"].values[0].strip('%'))
        self.assertLess(severe_change, mild_change)

if __name__ == "__main__":
    unittest.main() 