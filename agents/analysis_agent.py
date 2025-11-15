from crewai import Agent
import pandas as pd
import numpy as np
from datetime import datetime
import logging


class AnalysisAgent(Agent):
    """
    Agent responsible for analyzing sustainability data and identifying trends,
    anomalies, and optimization opportunities.
    """

    def __init__(self):
        super().__init__(
            role='Sustainability Data Analyst',
            goal='Identify patterns, trends, and anomalies in environmental data',
            backstory='Environmental data scientist with expertise in healthcare sustainability'
        )
        # set logger as a runtime-only attribute (avoid pydantic model validation)
        object.__setattr__(self, 'logger', logging.getLogger(__name__))

    def run(self, inputs: dict) -> dict:
        """
        Perform comprehensive analysis on collected data.

        Args:
            inputs: Dictionary containing data from Data Collector Agent

        Returns:
            Analysis results with insights and recommendations
        """
        try:
            self.logger.info("Starting data analysis...")

            df = pd.DataFrame(inputs.get('data'))

            # Perform multiple analysis types
            analysis_results = {
                'summary_statistics': self.calculate_summary_stats(df),
                'trend_analysis': self.analyze_trends(df),
                'anomaly_detection': self.detect_anomalies(df),
                'facility_comparison': self.compare_facilities(df),
                'carbon_intensity': self.calculate_carbon_intensity(df),
                'efficiency_metrics': self.calculate_efficiency_metrics(df)
            }

            return {
                'status': 'success',
                'analysis': analysis_results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def calculate_summary_stats(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive summary statistics."""
        metrics = [
            'electricity_kwh', 'natural_gas_therms', 'water_gallons',
            'waste_kg', 'recycling_kg', 'carbon_emissions_kg'
        ]

        summary = {}
        for metric in metrics:
            if metric in df.columns:
                summary[metric] = {
                    'mean': float(df[metric].mean()),
                    'median': float(df[metric].median()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'total': float(df[metric].sum())
                }

        return summary

    def analyze_trends(self, df: pd.DataFrame) -> dict:
        """Identify trends over time."""
        if 'date' in df.columns:
            try:
                df = df.sort_values('date')
            except Exception:
                pass

        trends = {}
        metrics = ['electricity_kwh', 'carbon_emissions_kg', 'recycling_rate']

        for metric in metrics:
            if metric in df.columns:
                first_value = df[metric].iloc[0]
                last_value = df[metric].iloc[-1]
                pct_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0

                trends[metric] = {
                    'direction': 'increasing' if pct_change > 0 else 'decreasing',
                    'percentage_change': round(pct_change, 2),
                    'trend_strength': 'strong' if abs(pct_change) > 10 else 'moderate' if abs(pct_change) > 5 else 'weak'
                }

        return trends

    def detect_anomalies(self, df: pd.DataFrame) -> dict:
        """Detect anomalies using statistical methods."""
        anomalies = {}

        for col in ['electricity_kwh', 'carbon_emissions_kg', 'water_gallons']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()

                upper_threshold = mean + (2 * std)
                lower_threshold = mean - (2 * std)

                anomaly_indices = df[(df[col] > upper_threshold) | (df[col] < lower_threshold)].index.tolist()

                anomalies[col] = {
                    'count': len(anomaly_indices),
                    'percentage': round(len(anomaly_indices) / len(df) * 100, 2) if len(df) > 0 else 0,
                    'threshold_upper': round(upper_threshold, 2),
                    'threshold_lower': round(lower_threshold, 2)
                }

        return anomalies

    def compare_facilities(self, df: pd.DataFrame) -> dict:
        """Compare performance across facilities."""
        if 'facility_id' not in df.columns:
            return {}

        comparison = df.groupby('facility_id').agg({
            'electricity_kwh': 'mean',
            'carbon_emissions_kg': 'mean',
            'recycling_rate': 'mean'
        }).round(2).to_dict()

        return comparison

    def calculate_carbon_intensity(self, df: pd.DataFrame) -> dict:
        """Calculate carbon emissions per unit of energy."""
        if 'carbon_emissions_kg' in df.columns and 'total_energy_mwh' in df.columns:
            df['carbon_intensity'] = df['carbon_emissions_kg'] / df['total_energy_mwh']

            return {
                'average_intensity': round(df['carbon_intensity'].mean(), 2),
                'best_performance': round(df['carbon_intensity'].min(), 2),
                'worst_performance': round(df['carbon_intensity'].max(), 2)
            }
        return {}

    def calculate_efficiency_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate various efficiency metrics."""
        metrics = {}

        if 'electricity_kwh' in df.columns:
            metrics['energy_intensity_kwh_per_sqft'] = round(df['electricity_kwh'].mean() / 100000, 4)

        if 'recycling_kg' in df.columns and 'waste_kg' in df.columns:
            total_waste = df['waste_kg'] + df['recycling_kg']
            metrics['waste_diversion_rate'] = round((df['recycling_kg'].sum() / total_waste.sum()) * 100, 2)

        if 'water_gallons' in df.columns:
            metrics['water_usage_per_day'] = round(df['water_gallons'].mean(), 2)

        return metrics
