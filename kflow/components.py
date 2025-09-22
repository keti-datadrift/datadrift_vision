from kfp import dsl
import requests

@dsl.component
def drift_check_component(api_url: str, period: str, event_name: str, threshold: float):
    response = requests.get(
        f"{api_url}/api/db_check_drift/",
        params={
            "period": period,
            "event_name": event_name,
            "threshold": threshold
        }
    )
    print("Drift check response:", response.json())
    return response.json()
