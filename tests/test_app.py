import pytest

from main import app
from source.ml import load_model
from source import config


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_app(client):
    response = client.get('/')
    assert '200' in str(response), str(response)

def test_load_model():
    load_model(config.MODEL_NAME)
