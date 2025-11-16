import logging
from datetime import datetime
import json
import os

# Import agents (may initialize LLMs) - we will instantiate them lazily and
# fall back to pure-Python local implementations if LLM/provider is not available.
try:
    from agents.data_collector import DataCollectorAgent
    from agents.analysis_agent import AnalysisAgent
    from agents.report_generator import ReportGeneratorAgent
    from agents.intervention_planner import InterventionPlannerAgent
except Exception:
    # Import errors will be handled at instantiation time; continue.
    DataCollectorAgent = None
    AnalysisAgent = None
    ReportGeneratorAgent = None
    InterventionPlannerAgent = None

# Import utilities
from utils.memory import SessionMemory, LongTermMemory
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/workflow.log'),
        logging.StreamHandler()
    ]
)


class SustainabilityAgentOrchestrator:
    """
    Main orchestrator coordinating all agents in the sustainability workflow.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session_memory = SessionMemory()
        self.long_term_memory = LongTermMemory()

        # Agents are created lazily in _ensure_agents so we can fall back to
        # local no-LLM implementations when running in restricted environments
        # (e.g., Kaggle kernels) or when OPENAI_API_KEY is missing.
        self.data_collector = None
        self.analyzer = None
        self.report_generator = None
        self.intervention_planner = None

        self.logger.info("Orchestrator initialized (agents will be created lazily)")

    def _ensure_agents(self):
        """Instantiate agent implementations, preferring CrewAI-based agents when
        available and enabled, otherwise fall back to lightweight local classes.
        """
        if self.data_collector is not None:
            return

        enable_llm = os.getenv('ENABLE_LLM', '1')

        # Allow selecting an LLM provider via environment variable. Supported
        # providers map to specific env var names for API keys. We do NOT set
        # any API keys in code; the user must provide them in their environment.
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        provider_key_map = {
            'openai': 'OPENAI_API_KEY',
            'groq': 'GROQ_API_KEY'
        }
        provider_key_env = provider_key_map.get(provider)

        # Local lightweight fallbacks
        class LocalDataCollector:
            def __init__(self, logger=None):
                self.logger = logger or logging.getLogger('LocalDataCollector')

            def run(self, inputs: dict):
                try:
                    data_path = inputs.get('data_path', 'data/hospital_energy.csv')
                    df = pd.read_csv(data_path)

                    rename_map = {}
                    if 'facility_id' in df.columns and 'facility' not in df.columns:
                        rename_map['facility_id'] = 'facility'
                    if 'facility_name' in df.columns and 'facility' not in rename_map and 'facility' not in df.columns:
                        rename_map['facility_name'] = 'facility'
                    if rename_map:
                        df = df.rename(columns=rename_map)

                    if 'energy_usage_kwh' in df.columns and 'electricity_kwh' not in df.columns:
                        df['electricity_kwh'] = df['energy_usage_kwh']
                    if 'emissions_kgco2' in df.columns and 'carbon_emissions_kg' not in df.columns:
                        df['carbon_emissions_kg'] = df['emissions_kgco2']

                    # Basic enrich: convert water liters if present
                    if 'water_usage_liters' in df.columns and 'water_gallons' not in df.columns:
                        df['water_gallons'] = df['water_usage_liters'] * 0.264172

                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

                    numeric_columns = [
                        'electricity_kwh',
                        'energy_usage_kwh',
                        'carbon_emissions_kg',
                        'emissions_kgco2',
                        'water_usage_liters',
                        'water_gallons',
                        'operational_hours'
                    ]
                    for column in numeric_columns:
                        if column in df.columns:
                            df[column] = pd.to_numeric(df[column], errors='coerce')

                    self.logger.info(f"Loaded {len(df)} records from {data_path}")
                    return {
                        'status': 'success',
                        'data': df.to_dict(orient='list'),
                        'validation': {'total_records': len(df)},
                        'timestamp': datetime.now().isoformat(),
                        'record_count': len(df)
                    }
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}

        class LocalAnalysisAgent:
            def __init__(self, logger=None):
                self.logger = logger or logging.getLogger('LocalAnalysisAgent')

            def run(self, inputs: dict):
                try:
                    df = pd.DataFrame(inputs.get('data'))
                    result = {'summary_statistics': {}, 'trend_analysis': {}, 'anomaly_detection': {}, 'facility_comparison': {}, 'carbon_intensity': {}, 'efficiency_metrics': {}}

                    # Summary stats for some common metrics
                    for metric in ['electricity_kwh', 'carbon_emissions_kg', 'water_gallons']:
                        if metric in df.columns:
                            result['summary_statistics'][metric] = {
                                'mean': float(df[metric].mean()),
                                'median': float(df[metric].median()),
                                'std': float(df[metric].std()),
                                'min': float(df[metric].min()),
                                'max': float(df[metric].max()),
                                'total': float(df[metric].sum())
                            }

                    # Simple trend: compare first and last
                    if 'date' in df.columns:
                        try:
                            df = df.sort_values('date')
                        except Exception:
                            pass
                    for metric in ['electricity_kwh', 'carbon_emissions_kg']:
                        if metric in df.columns and len(df) > 1:
                            first = df[metric].iloc[0]
                            last = df[metric].iloc[-1]
                            pct = ((last - first) / first * 100) if first != 0 else 0
                            result['trend_analysis'][metric] = {'direction': 'increasing' if pct>0 else 'decreasing', 'percentage_change': round(pct,2)}

                    # Anomaly detection (2 sigma)
                    for metric in ['electricity_kwh', 'carbon_emissions_kg', 'water_gallons']:
                        if metric in df.columns:
                            mean = df[metric].mean(); std = df[metric].std()
                            upper = mean + 2*std; lower = mean - 2*std
                            anomalies = df[(df[metric]>upper)|(df[metric]<lower)]
                            result['anomaly_detection'][metric] = {'count': int(len(anomalies)), 'percentage': round(len(anomalies)/len(df)*100,2) if len(df)>0 else 0, 'threshold_upper': float(upper), 'threshold_lower': float(lower)}

                    # Efficiency metrics
                    if 'electricity_kwh' in df.columns:
                        result['efficiency_metrics']['energy_intensity_kwh_per_sqft'] = round(df['electricity_kwh'].mean()/100000,4)
                    if 'water_gallons' in df.columns:
                        result['efficiency_metrics']['water_usage_per_day'] = round(df['water_gallons'].mean(),2)

                    return {'status': 'success', 'analysis': result, 'timestamp': datetime.now().isoformat()}
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}

        class LocalReportGenerator:
            def __init__(self, logger=None):
                self.logger = logger or logging.getLogger('LocalReportGenerator')
                try:
                    plt.style.use('seaborn')
                except Exception:
                    try:
                        plt.style.use('ggplot')
                    except Exception:
                        pass

            def run(self, inputs: dict):
                try:
                    analysis = inputs.get('analysis', {})
                    raw = inputs.get('raw_data')
                    report_text = 'Report generated (local fallback)'
                    viz_paths = []
                    if raw is not None:
                        df = pd.DataFrame(raw)
                        os.makedirs('outputs', exist_ok=True)
                        if 'date' in df.columns and 'electricity_kwh' in df.columns:
                            plt.figure(figsize=(10,5))
                            plt.plot(pd.to_datetime(df['date']), df['electricity_kwh'])
                            path = 'outputs/energy_consumption_trend.png'
                            plt.savefig(path, bbox_inches='tight')
                            plt.close()
                            viz_paths.append(path)
                    exec_summary = analysis.get('summary_statistics', {})
                    return {'status':'success','report':report_text,'executive_summary':str(exec_summary),'visualizations':viz_paths,'timestamp':datetime.now().isoformat()}
                except Exception as e:
                    return {'status':'error','message':str(e)}

        class LocalInterventionPlanner:
            def __init__(self, logger=None):
                self.logger = logger or logging.getLogger('LocalInterventionPlanner')

            def run(self, inputs: dict):
                try:
                    recommendations = [
                        {'title':'Enhance Recycling Program','category':'Waste Management','impact':'Medium','cost':'Low','timeframe':'1-3 months','description':'Implement waste segregation and training','estimated_savings':'30-50% increase in waste diversion'},
                        {'title':'Launch Green Team Initiative','category':'Staff Engagement','impact':'Medium','cost':'Low','timeframe':'Ongoing','description':'Create sustainability team','estimated_savings':'5-10% reduction across metrics'}
                    ]
                    roadmap = {'Phase 1 (0-3 months) - Quick Wins':[r['title'] for r in recommendations],'Phase 2 (3-6 months) - Medium-term':[],'Phase 3 (6-12 months) - Long-term':[]}
                    return {'status':'success','recommendations':recommendations,'implementation_roadmap':roadmap,'timestamp':datetime.now().isoformat()}
                except Exception as e:
                    return {'status':'error','message':str(e)}

        # Decide whether to use CrewAI agents or local fallbacks
        # Decide whether to use CrewAI agents or local fallbacks
        use_llm = enable_llm == '1' and DataCollectorAgent is not None
        if use_llm:
            # If a provider key env var is defined for the selected provider,
            # require it to be present before attempting initialization.
            if provider_key_env and provider_key_env not in os.environ:
                self.logger.warning(f"LLM provider '{provider}' selected but required env var '{provider_key_env}' is missing; falling back to local implementations")
                use_llm = False

        if use_llm:
            try:
                self.data_collector = DataCollectorAgent()
                self.analyzer = AnalysisAgent()
                self.report_generator = ReportGeneratorAgent()
                self.intervention_planner = InterventionPlannerAgent()
                self.logger.info('Using CrewAI-powered agents')
                return
            except Exception as e:
                self.logger.warning(f'Failed to initialize CrewAI agents, falling back to local implementations: {e}')

        # Fallback to local implementations
        self.data_collector = LocalDataCollector(self.logger)
        self.analyzer = LocalAnalysisAgent(self.logger)
        self.report_generator = LocalReportGenerator(self.logger)
        self.intervention_planner = LocalInterventionPlanner(self.logger)
        self.logger.info('Using local fallback agents (LLM disabled or unavailable)')

    def run_full_workflow(self, data_path='data/hospital_energy.csv'):
        """
        Execute complete sustainability analysis workflow.

        Args:
            data_path: Path to input data file

        Returns:
            Complete workflow results
        """
        workflow_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.info(f"Starting workflow {workflow_id}")

        # Ensure agents are available (either CrewAI-based or local fallbacks)
        try:
            self._ensure_agents()
        except Exception as e:
            self.logger.error(f"Failed to prepare agents: {e}")
            return {
                'workflow_id': workflow_id,
                'status': 'error',
                'message': str(e),
                'session_history': self.session_memory.get_history()
            }

        try:
            # Step 1: Data Collection
            self.logger.info("Step 1/4: Collecting data...")
            self.session_memory.add_to_history("Data collection started")

            collection_result = self.data_collector.run({'data_path': data_path})

            if collection_result['status'] != 'success':
                raise Exception(f"Data collection failed: {collection_result.get('message')}")

            self.session_memory.save('collected_data', collection_result)
            self.logger.info(f"[OK] Collected {collection_result['record_count']} records")

            # Step 2: Analysis
            self.logger.info("Step 2/4: Analyzing data...")
            self.session_memory.add_to_history("Analysis started")

            analysis_result = self.analyzer.run(collection_result)

            if analysis_result['status'] != 'success':
                raise Exception(f"Analysis failed: {analysis_result.get('message')}")

            self.session_memory.save('analysis_results', analysis_result)
            self.logger.info("[OK] Analysis completed")

            # Step 3: Report Generation
            self.logger.info("Step 3/4: Generating reports...")
            self.session_memory.add_to_history("Report generation started")

            report_input = {
                'analysis': analysis_result['analysis'],
                'raw_data': collection_result['data']
            }
            report_result = self.report_generator.run(report_input)

            if report_result['status'] != 'success':
                raise Exception(f"Report generation failed: {report_result.get('message')}")

            self.session_memory.save('report', report_result)
            self.logger.info(f"[OK] Generated {len(report_result.get('visualizations', []))} visualizations")

            # Step 4: Intervention Planning
            self.logger.info("Step 4/4: Planning interventions...")
            self.session_memory.add_to_history("Intervention planning started")

            intervention_result = self.intervention_planner.run({'analysis': analysis_result['analysis']})

            if intervention_result['status'] != 'success':
                raise Exception(f"Intervention planning failed: {intervention_result.get('message')}")

            self.session_memory.save('interventions', intervention_result)
            self.logger.info(f"[OK] Generated {len(intervention_result['recommendations'])} recommendations")

            # Save to long-term memory
            self.long_term_memory.remember(
                f'workflow_{workflow_id}',
                {
                    'analysis': analysis_result,
                    'report': report_result,
                    'interventions': intervention_result
                }
            )

            # Compile final results
            final_results = {
                'workflow_id': workflow_id,
                'status': 'success',
                'data_collection': collection_result,
                'analysis': analysis_result,
                'report': report_result,
                'interventions': intervention_result,
                'session_history': self.session_memory.get_history()
            }

            self.logger.info(f"[OK] Workflow {workflow_id} completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            return {
                'workflow_id': workflow_id,
                'status': 'error',
                'message': str(e),
                'session_history': self.session_memory.get_history()
            }

    def display_results(self, results):
        """Pretty print workflow results."""
        if results['status'] == 'error':
            print(f"\n❌ Workflow Failed: {results['message']}")
            return

        print("\n" + "="*80)
        print("HEALTHCARE SUSTAINABILITY ANALYSIS - COMPLETE RESULTS")
        print("="*80)

        # Executive Summary
        if 'report' in results and 'executive_summary' in results['report']:
            print("\n" + results['report']['executive_summary'])

        # Detailed Report
        if 'report' in results and 'report' in results['report']:
            print("\n" + results['report']['report'])

        # Recommendations
        if 'interventions' in results and 'recommendations' in results['interventions']:
            print("\n" + "="*80)
            print("RECOMMENDED INTERVENTIONS")
            print("="*80 + "\n")

            for i, rec in enumerate(results['interventions']['recommendations'], 1):
                print(f"{i}. {rec['title']}")
                print(f"   Category: {rec['category']}")
                print(f"   Impact: {rec['impact']} | Cost: {rec['cost']} | Timeline: {rec['timeframe']}")
                print(f"   {rec['description']}")
                print(f"   Expected Savings: {rec['estimated_savings']}")
                print()

        # Implementation Roadmap
        if 'interventions' in results and 'implementation_roadmap' in results['interventions']:
            print("\n" + "="*80)
            print("IMPLEMENTATION ROADMAP")
            print("="*80 + "\n")

            roadmap = results['interventions']['implementation_roadmap']
            for phase, items in roadmap.items():
                print(f"\n{phase}:")
                for item in items:
                    print(f"  • {item}")

        print("\n" + "="*80)
        print(f"Analysis completed at: {results.get('workflow_id', 'N/A')}")
        print("="*80 + "\n")
