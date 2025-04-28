import requests


def test_app():
    """
    Test the health check endpoint of the Flask application.
    """

    base_url = "http://localhost:5000"
    response = requests.get(f"{base_url}/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}