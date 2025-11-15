from crewai import Agent
import pandas as pd
import logging
from datetime import datetime


class DataCollectorAgent(Agent):
    """
    Agent responsible for collecting sustainability data from various sources.
    Handles CSV files, APIs, and data validation.
    """

    def __init__(self):
        super().__init__(
            role='Data Collection Specialist',
            goal='Collect comprehensive sustainability metrics from healthcare facilities',
            backstory='Expert in data extraction and validation for environmental monitoring'
        )
        # set logger as a runtime-only attribute (avoid pydantic model validation)
        object.__setattr__(self, 'logger', logging.getLogger(__name__))

    def run(self, inputs: dict):
        """
        Main execution method for data collection.

        Args:
            inputs: Dictionary containing data source paths and parameters

        Returns:
            Dictionary with collected data and metadata
        """
        try:
            self.logger.info("Starting data collection process...")

            # Load primary dataset
            data_path = inputs.get('data_path', 'data/hospital_energy.csv')
            data = self.load_csv_data(data_path)

            # Validate data quality
            validation_report = self.validate_data(data)

            # Enrich with additional metadata
            enriched_data = self.enrich_data(data)

            return {
                'status': 'success',
                'data': enriched_data.to_dict(orient='list'),
                'validation': validation_report,
                'timestamp': datetime.now().isoformat(),
                'record_count': len(enriched_data)
            }

        except Exception as e:
            self.logger.error(f"Data collection failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """Load and parse CSV data."""
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded {len(df)} records from {filepath}")
        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality and completeness.

        Returns validation report with issues found.
        """
        report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': int(df.duplicated().sum()),
        }

        # If date column exists, include range
        if 'date' in df.columns:
            try:
                report['date_range'] = {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max())
                }
            except Exception:
                pass

        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        negative_checks = {}
        for col in numeric_cols:
            negative_count = int((df[col] < 0).sum())
            if negative_count > 0:
                negative_checks[col] = negative_count

        report['negative_values'] = negative_checks
        return report

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields and metadata."""
        df = df.copy()

        # Convert date to datetime when possible
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception:
                pass

        # Add day of week if date present
        if 'date' in df.columns:
            try:
                df['day_of_week'] = df['date'].dt.day_name()
            except Exception:
                pass

        # Normalize common column name variants from different data sources
        # and perform simple unit conversions where appropriate.
        # This helps downstream agents which expect canonical names.
        aliases = {
            'energy_usage_kwh': 'electricity_kwh',
            'emissions_kgco2': 'carbon_emissions_kg',
            'water_usage_liters': 'water_gallons'
        }

        # Convert common units and create canonical columns without causing duplicates.
        # Handle water liters -> gallons conversion first (if present).
        try:
            if 'water_usage_liters' in df.columns and 'water_gallons' not in df.columns:
                df['water_gallons'] = df['water_usage_liters'] * 0.264172
                # drop original liters column to avoid duplicate semantics
                df = df.drop(columns=['water_usage_liters'])
        except Exception:
            pass

        # Apply renames for any alias columns found, but only when the target
        # canonical name does not already exist (avoid duplicate column names).
        rename_map = {k: v for k, v in aliases.items() if k in df.columns and v not in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        # Calculate a normalized total energy if fields exist
        if 'electricity_kwh' in df.columns and 'natural_gas_therms' in df.columns:
            df['total_energy_mwh'] = (df['electricity_kwh'] / 1000) + (df['natural_gas_therms'] * 0.029307)

        # Calculate recycling rate where applicable
        if 'recycling_kg' in df.columns and 'waste_kg' in df.columns:
            total_waste = df['waste_kg'] + df['recycling_kg']
            df['recycling_rate'] = (df['recycling_kg'] / total_waste.replace({0: pd.NA})) * 100

        return df
