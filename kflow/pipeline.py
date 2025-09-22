from components import *

@dsl.pipeline(
    name="Data Drift Detection Pipeline",
    description="Periodically checks for data drift"
)
def drift_pipeline(
    api_url: str = "http://fastapi-service.default.svc.cluster.local:8000",
    period: str = "1 day",
    event_name: str = "실시간 인식",
    threshold: float = 0.3
):
    drift_check_component(api_url, period, event_name, threshold)
