from crewai import Agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import logging
import os


class ReportGeneratorAgent(Agent):
    """
    Agent responsible for generating comprehensive sustainability reports
    with visualizations and executive summaries.
    """

    def __init__(self):
        super().__init__(
            role='Sustainability Report Specialist',
            goal='Create clear, actionable sustainability reports with visualizations',
            backstory='Communication expert specializing in environmental reporting'
        )
        # set logger as a runtime-only attribute (avoid pydantic model validation)
        object.__setattr__(self, 'logger', logging.getLogger(__name__))
        sns.set_style("whitegrid")

    def run(self, inputs: dict) -> dict:
        """
        Generate comprehensive sustainability report.

        Args:
            inputs: Dictionary containing analysis results

        Returns:
            Report data and visualization paths
        """
        try:
            self.logger.info("Generating sustainability report...")

            analysis = inputs.get('analysis', {})
            raw_data = inputs.get('raw_data')

            # Generate text report
            text_report = self.create_text_report(analysis)

            # Generate visualizations
            viz_paths = self.create_visualizations(raw_data, analysis)

            # Create executive summary
            exec_summary = self.create_executive_summary(analysis)

            return {
                'status': 'success',
                'report': text_report,
                'executive_summary': exec_summary,
                'visualizations': viz_paths,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def create_executive_summary(self, analysis: dict) -> str:
        """Create high-level executive summary."""
        summary = "=== EXECUTIVE SUMMARY ===\n\n"

        # Carbon emissions summary
        if 'summary_statistics' in analysis:
            carbon_stats = analysis['summary_statistics'].get('carbon_emissions_kg', {})
            summary += f"Total Carbon Emissions: {carbon_stats.get('total', 0):,.0f} kg CO2\n"
            summary += f"Average Daily Emissions: {carbon_stats.get('mean', 0):,.0f} kg CO2\n\n"

        # Trends summary
        if 'trend_analysis' in analysis:
            trends = analysis['trend_analysis']
            summary += "Key Trends:\n"
            for metric, trend_data in trends.items():
                direction = trend_data.get('direction', 'stable')
                change = trend_data.get('percentage_change', 0)
                summary += f"  • {metric}: {direction.capitalize()} by {abs(change):.1f}%\n"
            summary += "\n"

        # Efficiency metrics
        if 'efficiency_metrics' in analysis:
            eff = analysis['efficiency_metrics']
            summary += "Efficiency Metrics:\n"
            for metric, value in eff.items():
                summary += f"  • {metric.replace('_', ' ').title()}: {value}\n"

        return summary

    def create_text_report(self, analysis: dict) -> str:
        """Create detailed text-based report."""
        report = "=" * 60 + "\n"
        report += "HEALTHCARE FACILITY SUSTAINABILITY REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 60 + "\n\n"

        # Summary Statistics Section
        if 'summary_statistics' in analysis:
            report += "--- SUMMARY STATISTICS ---\n\n"
            for metric, stats in analysis['summary_statistics'].items():
                report += f"{metric.replace('_', ' ').title()}:\n"
                report += f"  Mean: {stats.get('mean', 0):,.2f}\n"
                report += f"  Median: {stats.get('median', 0):,.2f}\n"
                report += f"  Range: {stats.get('min', 0):,.2f} - {stats.get('max', 0):,.2f}\n"
                report += f"  Total: {stats.get('total', 0):,.2f}\n\n"

        # Anomaly Detection Section
        if 'anomaly_detection' in analysis:
            report += "--- ANOMALY DETECTION ---\n\n"
            for metric, anomaly_data in analysis['anomaly_detection'].items():
                count = anomaly_data.get('count', 0)
                if count > 0:
                    report += f"{metric}: {count} anomalies detected "
                    report += f"({anomaly_data.get('percentage', 0)}% of data)\n"
            report += "\n"

        # Carbon Intensity Section
        if 'carbon_intensity' in analysis:
            report += "--- CARBON INTENSITY ANALYSIS ---\n\n"
            ci = analysis['carbon_intensity']
            report += f"Average: {ci.get('average_intensity', 0):.2f} kg CO2/MWh\n"
            report += f"Best Performance: {ci.get('best_performance', 0):.2f} kg CO2/MWh\n"
            report += f"Worst Performance: {ci.get('worst_performance', 0):.2f} kg CO2/MWh\n\n"

        return report

    def create_visualizations(self, raw_data, analysis: dict) -> list:
        """Create and save visualization charts."""
        viz_paths = []

        if raw_data is None:
            return viz_paths

        df = pd.DataFrame(raw_data)

        # Ensure output directory exists
        os.makedirs('outputs', exist_ok=True)

        # Visualization 1: Energy Consumption Over Time
        if 'date' in df.columns and 'electricity_kwh' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(df['date']), df['electricity_kwh'], marker='o', linewidth=2)
            plt.title('Electricity Consumption Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Electricity (kWh)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            path1 = 'outputs/energy_consumption_trend.png'
            plt.savefig(path1, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(path1)

        # Visualization 2: Carbon Emissions by Facility
        if 'facility_id' in df.columns and 'carbon_emissions_kg' in df.columns:
            plt.figure(figsize=(10, 6))
            facility_emissions = df.groupby('facility_id')['carbon_emissions_kg'].sum()
            facility_emissions.plot(kind='bar', color='steelblue')
            plt.title('Total Carbon Emissions by Facility', fontsize=16, fontweight='bold')
            plt.xlabel('Facility ID', fontsize=12)
            plt.ylabel('Carbon Emissions (kg CO2)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            path2 = 'outputs/emissions_by_facility.png'
            plt.savefig(path2, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(path2)

        # Visualization 3: Recycling Rate Distribution
        if 'recycling_rate' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['recycling_rate'], bins=20, color='green', alpha=0.7, edgecolor='black')
            plt.title('Recycling Rate Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Recycling Rate (%)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.axvline(df['recycling_rate'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["recycling_rate"].mean():.1f}%')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            path3 = 'outputs/recycling_distribution.png'
            plt.savefig(path3, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(path3)

        self.logger.info(f"Generated {len(viz_paths)} visualizations")
        return viz_paths
