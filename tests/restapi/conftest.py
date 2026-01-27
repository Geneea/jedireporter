from typing import Any, Generator

import pytest
from starlette.testclient import TestClient

from jedireporter.restapi import server


@pytest.fixture()
def client() -> Generator[TestClient, None, Any]:
    with TestClient(server.create_app()) as client:
        yield client
