from main_workflow import SustainabilityAgentOrchestrator


def test_full_workflow():
    """Test complete agent workflow"""
    orchestrator = SustainabilityAgentOrchestrator()
    results = orchestrator.run_full_workflow('data/hospital_energy.csv')

    assert results['status'] == 'success', "Workflow should complete successfully"
    assert 'analysis' in results, "Should contain analysis results"
    assert 'report' in results, "Should contain report"
    assert 'interventions' in results, "Should contain recommendations"

    print("✓ All integration tests passed")


if __name__ == '__main__':
    test_full_workflow()
