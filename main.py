from main_workflow import SustainabilityAgentOrchestrator


def main():
    orchestrator = SustainabilityAgentOrchestrator()
    results = orchestrator.run_full_workflow()
    orchestrator.display_results(results)


if __name__ == "__main__":
    main()

