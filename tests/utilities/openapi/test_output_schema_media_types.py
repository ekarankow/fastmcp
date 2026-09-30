"""Output schemas are only derived from success responses that declare JSON documents."""

from typing import Any

import pytest

from fastmcp.utilities.openapi.models import ResponseInfo
from fastmcp.utilities.openapi.schemas import extract_output_schema_from_responses

BINARY = {"type": "string", "format": "binary"}
ITEM = {"type": "object", "properties": {"name": {"type": "string"}}}


def _output_schema(content: dict[str, Any]) -> dict[str, Any] | None:
    return extract_output_schema_from_responses(
        {"200": ResponseInfo(description="OK", content_schema=content)},
        openapi_version="3.0.3",
    )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param({"application/zip": BINARY}, id="zip"),
        pytest.param({"image/png": BINARY}, id="image"),
        pytest.param({"text/markdown": {"type": "string"}}, id="text"),
        pytest.param({"application/octet-stream": BINARY}, id="octet-stream"),
        pytest.param({"application/json": BINARY}, id="json-declared-binary"),
        pytest.param(
            {
                "application/json": {
                    "type": "string",
                    "contentMediaType": "application/octet-stream",
                }
            },
            id="json-declared-binary-3.1",
        ),
    ],
)
def test_non_json_success_response_has_no_output_schema(content: dict[str, Any]):
    assert _output_schema(content) is None


@pytest.mark.parametrize(
    "content",
    [
        pytest.param({"application/json": ITEM}, id="json"),
        pytest.param({"application/hal+json": ITEM}, id="json-compatible"),
        pytest.param({"application/vnd.acme+json": ITEM}, id="vendor-json"),
        pytest.param({"application/zip": BINARY, "application/json": ITEM}, id="mixed"),
    ],
)
def test_json_success_response_keeps_output_schema(content: dict[str, Any]):
    schema = _output_schema(content)

    assert schema is not None
    assert schema["properties"]["name"] == {"type": "string"}


def test_json_string_response_is_still_wrapped():
    schema = _output_schema({"application/json": {"type": "string"}})

    assert schema is not None
    assert schema["x-fastmcp-wrap-result"] is True
    assert schema["properties"]["result"] == {"type": "string"}
