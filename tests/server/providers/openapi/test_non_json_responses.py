"""OpenAPI tools return non-JSON success bodies as text or binary content."""

import base64
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI, Response
from mcp_types import EmbeddedResource, ImageContent, TextContent

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

ZIP_BYTES = b"PK\x03\x04\x14\x00\x00\x00\xff\xfe"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff"
MARKDOWN = "# Title\n\nSome *text*.\n"


def _binary(media_type: str) -> dict[str, Any]:
    return {media_type: {"schema": {"type": "string", "format": "binary"}}}


def _spec(openapi_version: str) -> dict[str, Any]:
    def op(operation_id: str, content: dict[str, Any] | None) -> dict[str, Any]:
        response: dict[str, Any] = {"description": "OK"}
        if content is not None:
            response["content"] = content
        return {"get": {"operationId": operation_id, "responses": {"200": response}}}

    return {
        "openapi": openapi_version,
        "info": {"title": "Files", "version": "1"},
        "paths": {
            "/archive": op("get_archive", _binary("application/zip")),
            "/image": op("get_image", _binary("image/png")),
            "/readme": op(
                "get_readme", {"text/markdown": {"schema": {"type": "string"}}}
            ),
            # A common mismatch: the file is declared as a JSON binary string but
            # served with its own media type.
            "/document": op("get_document", _binary("application/json")),
            "/blob": op("get_blob", _binary("application/octet-stream")),
            "/item": op(
                "get_item",
                {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        }
                    }
                },
            ),
            "/missing": op("get_missing", None),
        },
    }


def _app(blob: bytes) -> FastAPI:
    app = FastAPI()

    @app.get("/archive")
    async def archive():
        return Response(ZIP_BYTES, media_type="application/zip")

    @app.get("/image")
    async def image():
        return Response(PNG_BYTES, media_type="image/png")

    @app.get("/readme")
    async def readme():
        return Response(MARKDOWN, media_type="text/markdown")

    @app.get("/document")
    async def document():
        return Response(MARKDOWN, media_type="text/markdown")

    @app.get("/blob")
    async def blob_route():
        return Response(blob, media_type="application/octet-stream")

    @app.get("/item")
    async def item():
        return {"name": "a"}

    @app.get("/missing")
    async def missing():
        return Response(
            '{"detail": "gone"}', status_code=404, media_type="application/json"
        )

    return app


async def _call(
    openapi_version: str, tool: str, blob: bytes = b""
) -> tuple[Any, dict[str, Any]]:
    async with httpx2.AsyncClient(
        base_url="http://test", transport=httpx2.ASGITransport(app=_app(blob))
    ) as http:
        server = FastMCP.from_openapi(openapi_spec=_spec(openapi_version), client=http)
        async with Client(server) as client:
            tools = {t.name: t for t in await client.list_tools()}
            result = await client.call_tool(tool, {})
    return result, tools


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
async def test_binary_body_is_returned_as_blob_resource(openapi_version: str):
    result, tools = await _call(openapi_version, "get_archive")

    (resource,) = [c for c in result.content if isinstance(c, EmbeddedResource)]
    assert resource.resource.mime_type == "application/zip"
    assert base64.b64decode(resource.resource.blob) == ZIP_BYTES
    assert str(resource.resource.uri) == "http://test/archive"
    assert tools["get_archive"].output_schema is None


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
async def test_image_body_is_returned_as_image_content(openapi_version: str):
    result, _ = await _call(openapi_version, "get_image")

    (image,) = [c for c in result.content if isinstance(c, ImageContent)]
    assert image.mime_type == "image/png"
    assert base64.b64decode(image.data) == PNG_BYTES


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
@pytest.mark.parametrize("tool", ["get_readme", "get_document"])
async def test_text_body_is_returned_as_text(openapi_version: str, tool: str):
    result, tools = await _call(openapi_version, tool)

    assert [c.text for c in result.content if isinstance(c, TextContent)] == [MARKDOWN]
    assert tools[tool].output_schema is None


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
async def test_untyped_utf8_body_is_returned_as_text(openapi_version: str):
    result, _ = await _call(openapi_version, "get_blob", blob=MARKDOWN.encode())

    assert [c.text for c in result.content if isinstance(c, TextContent)] == [MARKDOWN]


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
async def test_untyped_binary_body_stays_binary(openapi_version: str):
    data = b"\x00\x01\xffbinary"
    result, _ = await _call(openapi_version, "get_blob", blob=data)

    (resource,) = [c for c in result.content if isinstance(c, EmbeddedResource)]
    assert resource.resource.mime_type == "application/octet-stream"
    assert base64.b64decode(resource.resource.blob) == data


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
async def test_json_body_keeps_structured_content(openapi_version: str):
    result, tools = await _call(openapi_version, "get_item")

    assert result.structured_content == {"name": "a"}
    assert tools["get_item"].output_schema is not None


@pytest.mark.parametrize("openapi_version", ["3.0.3", "3.1.0"])
async def test_http_errors_still_raise(openapi_version: str):
    with pytest.raises(ToolError, match="404"):
        await _call(openapi_version, "get_missing")
