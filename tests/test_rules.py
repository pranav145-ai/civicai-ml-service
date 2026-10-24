from app.services.model_service import determine_urgency


def test_critical_water_health():
    urgency, _ = determine_urgency("Contaminated water caused illness and fever among residents.")
    assert urgency == "Critical"


def test_high_burst_pipeline():
    urgency, _ = determine_urgency("A burst water pipeline is flooding the shared residential street.")
    assert urgency == "High"


def test_medium_service_disruption():
    urgency, _ = determine_urgency("There is no water supply in our building since morning.")
    assert urgency == "Medium"


def test_ambiguous_case_needs_review():
    urgency, _ = determine_urgency("I have a concern about the public service.")
    assert urgency is None
