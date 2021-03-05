import pytest

from main import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_empty_db(client):
    response = client.get('/')
    assert '200' in str(response), str(response)
