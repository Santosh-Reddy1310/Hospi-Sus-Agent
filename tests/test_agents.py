import unittest
import pandas as pd
from agents.data_collector import DataCollectorAgent
from agents.analysis_agent import AnalysisAgent


class TestDataCollector(unittest.TestCase):

    def setUp(self):
        self.agent = DataCollectorAgent()

    def test_data_loading(self):
        """Test CSV data loading"""
        result = self.agent.run({'data_path': 'data/hospital_energy.csv'})
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['record_count'], 0)

    def test_data_validation(self):
        """Test data validation logic"""
        df = pd.DataFrame({
            'facility_id': ['A', 'B'],
            'electricity_kwh': [1000, 2000]
        })
        validation = self.agent.validate_data(df)
        self.assertIn('total_records', validation)
        self.assertEqual(validation['total_records'], 2)


class TestAnalysisAgent(unittest.TestCase):

    def setUp(self):
        self.agent = AnalysisAgent()

    def test_summary_statistics(self):
        """Test statistical calculations"""
        df = pd.DataFrame({
            'electricity_kwh': [1000, 1500, 2000]
        })
        stats = self.agent.calculate_summary_stats(df)
        self.assertIn('electricity_kwh', stats)
        self.assertEqual(stats['electricity_kwh']['mean'], 1500.0)


if __name__ == '__main__':
    unittest.main()
