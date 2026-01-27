import os
from typing import Any, Generator

import pytest
from starlette.testclient import TestClient

from jedireporter.restapi import server


@pytest.fixture()
def client() -> Generator[TestClient, None, Any]:
    if not os.getenv('JEDI_OPENAI_API_KEY'):
        os.environ['JEDI_OPENAI_API_KEY'] = 'test-openai-api-key'
    with TestClient(server.create_app()) as client:
        yield client
