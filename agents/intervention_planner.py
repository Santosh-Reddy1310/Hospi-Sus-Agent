from crewai import Agent
from datetime import datetime
import logging


class InterventionPlannerAgent(Agent):
    """
    Agent responsible for recommending sustainability interventions
    based on analysis results and industry best practices.
    """

    def __init__(self):
        super().__init__(
            role='Sustainability Intervention Strategist',
            goal='Develop actionable intervention plans to improve environmental performance',
            backstory='Healthcare sustainability consultant with expertise in operational improvements'
        )
        # set logger as a runtime-only attribute (avoid pydantic model validation)
        object.__setattr__(self, 'logger', logging.getLogger(__name__))
        # store thresholds as a runtime attribute
        object.__setattr__(self, 'thresholds', {
            'carbon_intensity': 500,  # kg CO2/MWh
            'recycling_rate': 40,  # percentage
            'energy_intensity': 0.15  # kWh per sq ft
        })

    def run(self, inputs: dict) -> dict:
        """
        Generate intervention recommendations based on analysis.

        Args:
            inputs: Dictionary containing analysis results

        Returns:
            Prioritized list of interventions with implementation details
        """
        try:
            self.logger.info("Generating intervention plan...")

            analysis = inputs.get('analysis', {})

            # Generate recommendations
            recommendations = self.generate_recommendations(analysis)

            # Prioritize interventions
            prioritized = self.prioritize_interventions(recommendations)

            # Create implementation roadmap
            roadmap = self.create_implementation_roadmap(prioritized)

            return {
                'status': 'success',
                'recommendations': prioritized,
                'implementation_roadmap': roadmap,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Intervention planning failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def generate_recommendations(self, analysis: dict) -> list:
        """Generate intervention recommendations based on analysis."""
        recommendations = []

        # Energy efficiency recommendations
        if 'summary_statistics' in analysis:
            elec_stats = analysis['summary_statistics'].get('electricity_kwh', {})
            avg_usage = elec_stats.get('mean', 0)

            if avg_usage > 15000:
                recommendations.append({
                    'category': 'Energy Efficiency',
                    'title': 'Implement LED Lighting Upgrade',
                    'description': 'Replace fluorescent and incandescent bulbs with LED fixtures',
                    'impact': 'High',
                    'estimated_savings': '20-30% reduction in lighting energy costs',
                    'cost': 'Medium',
                    'timeframe': '3-6 months',
                    'priority': 8
                })

                recommendations.append({
                    'category': 'Energy Efficiency',
                    'title': 'Optimize HVAC System Operations',
                    'description': 'Install smart thermostats and implement schedule-based controls',
                    'impact': 'High',
                    'estimated_savings': '15-25% reduction in heating/cooling costs',
                    'cost': 'Medium',
                    'timeframe': '2-4 months',
                    'priority': 9
                })

        # Carbon reduction recommendations
        if 'carbon_intensity' in analysis:
            avg_intensity = analysis['carbon_intensity'].get('average_intensity', 0)

            if avg_intensity > self.thresholds['carbon_intensity']:
                recommendations.append({
                    'category': 'Renewable Energy',
                    'title': 'Transition to Renewable Energy Sources',
                    'description': 'Purchase renewable energy credits or install on-site solar panels',
                    'impact': 'Very High',
                    'estimated_savings': '50-100% reduction in Scope 2 emissions',
                    'cost': 'High',
                    'timeframe': '6-12 months',
                    'priority': 10
                })

        # Waste management recommendations
        if 'efficiency_metrics' in analysis:
            recycling_rate = analysis['efficiency_metrics'].get('waste_diversion_rate', 0)

            if recycling_rate < self.thresholds['recycling_rate']:
                recommendations.append({
                    'category': 'Waste Management',
                    'title': 'Enhance Recycling Program',
                    'description': 'Implement comprehensive waste segregation and staff training',
                    'impact': 'Medium',
                    'estimated_savings': '30-50% increase in waste diversion',
                    'cost': 'Low',
                    'timeframe': '1-3 months',
                    'priority': 7
                })

        # Water conservation recommendations
        if 'summary_statistics' in analysis:
            water_stats = analysis['summary_statistics'].get('water_gallons', {})
            avg_water = water_stats.get('mean', 0)

            if avg_water > 45000:
                recommendations.append({
                    'category': 'Water Conservation',
                    'title': 'Install Low-Flow Fixtures',
                    'description': 'Replace faucets, toilets, and shower heads with water-efficient models',
                    'impact': 'Medium',
                    'estimated_savings': '20-35% reduction in water consumption',
                    'cost': 'Low',
                    'timeframe': '2-4 months',
                    'priority': 6
                })

        # Anomaly-based recommendations
        if 'anomaly_detection' in analysis:
            for metric, anomaly_data in analysis['anomaly_detection'].items():
                if anomaly_data.get('count', 0) > 2:
                    recommendations.append({
                        'category': 'Operational Efficiency',
                        'title': f'Investigate {metric} Anomalies',
                        'description': f'Review and address irregular patterns in {metric}',
                        'impact': 'Medium',
                        'estimated_savings': 'Variable - depends on root cause',
                        'cost': 'Low',
                        'timeframe': '1 month',
                        'priority': 5
                    })

        # Behavioral recommendations
        recommendations.append({
            'category': 'Staff Engagement',
            'title': 'Launch Green Team Initiative',
            'description': 'Create cross-functional sustainability team and awareness campaigns',
            'impact': 'Medium',
            'estimated_savings': '5-10% reduction across all metrics',
            'cost': 'Low',
            'timeframe': 'Ongoing',
            'priority': 6
        })

        return recommendations

    def prioritize_interventions(self, recommendations: list) -> list:
        """Sort recommendations by priority score."""
        return sorted(recommendations, key=lambda x: x['priority'], reverse=True)

    def create_implementation_roadmap(self, recommendations: list) -> dict:
        """Create phased implementation plan."""
        roadmap = {
            'Phase 1 (0-3 months) - Quick Wins': [],
            'Phase 2 (3-6 months) - Medium-term': [],
            'Phase 3 (6-12 months) - Long-term': []
        }

        for rec in recommendations:
            timeframe = rec.get('timeframe', '')

            if 'month' in timeframe:
                try:
                    months = int(''.join(filter(str.isdigit, timeframe.split('-')[0])))
                except Exception:
                    months = 0

                if months <= 3:
                    roadmap['Phase 1 (0-3 months) - Quick Wins'].append(rec['title'])
                elif months <= 6:
                    roadmap['Phase 2 (3-6 months) - Medium-term'].append(rec['title'])
                else:
                    roadmap['Phase 3 (6-12 months) - Long-term'].append(rec['title'])
            else:
                roadmap['Phase 1 (0-3 months) - Quick Wins'].append(rec['title'])

        return roadmap
