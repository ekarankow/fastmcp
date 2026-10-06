"""OpenAPI component classes: Tool, Resource, and ResourceTemplate."""

from __future__ import annotations

import base64
import json
import re
from typing import TYPE_CHECKING, Any

import httpx2
from mcp_types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ToolAnnotations,
)
from pydantic.networks import AnyUrl

from fastmcp.resources import (
    Resource,
    ResourceContent,
    ResourceResult,
    ResourceTemplate,
)
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.base import Tool, ToolResult
from fastmcp.utilities.exceptions import is_request_error, is_timeout_error
from fastmcp.utilities.logging import get_logger
from fastmcp.utilities.openapi import HTTPRoute
from fastmcp.utilities.openapi.director import RequestDirector
from fastmcp.utilities.tasks import TaskConfig

if TYPE_CHECKING:
    from fastmcp.server import Context

_SAFE_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "accept-language",
        "cache-control",
        "connection",
        "content-length",
        "content-type",
        "host",
        "user-agent",
    }
)


def _redact_headers(headers: httpx2.Headers) -> dict[str, str]:
    return {k: v if k.lower() in _SAFE_HEADERS else "***" for k, v in headers.items()}


__all__ = [
    "OpenAPIResource",
    "OpenAPIResourceTemplate",
    "OpenAPITool",
    "_extract_mime_type_from_route",
]

logger = get_logger(__name__)

# Default MIME type when no response content type can be inferred
_DEFAULT_MIME_TYPE = "application/json"


def _raise_for_status(response: httpx2.Response) -> None:
    """Raise an OpenAPI-formatted error without relying on client exception types.

    The error is an `httpx2.HTTPStatusError` whatever library the client
    belongs to, so the server can recognize an upstream error response by its
    status code.
    """
    if 200 <= response.status_code < 300:
        return

    error_message = f"HTTP error {response.status_code}: {response.reason_phrase}"
    try:
        error_data = response.json()
        error_message += f" - {error_data}"
    except (json.JSONDecodeError, ValueError):
        if response.text:
            error_message += f" - {response.text}"
    raise httpx2.HTTPStatusError(
        error_message, request=response.request, response=response
    )


async def _send_request(
    client: httpx2.AsyncClient,
    request: httpx2.Request,
) -> httpx2.Response:
    """Send a request while preserving transitional legacy-client errors."""
    try:
        return await client.send(request)
    except Exception as exc:
        if is_timeout_error(exc):
            raise ValueError(f"HTTP request timed out ({type(exc).__name__})") from exc
        if is_request_error(exc):
            raise ValueError(f"Request error ({type(exc).__name__}): {exc!s}") from exc
        raise


def _extract_mime_type_from_route(route: HTTPRoute) -> str:
    """Extract the primary MIME type from an HTTPRoute's response definitions.

    Looks for the first successful response (2xx) and returns its content type.
    Prefers JSON-compatible types when multiple are available.
    Falls back to "application/json" when no response content type is declared.
    """
    if not route.responses:
        return _DEFAULT_MIME_TYPE

    # Priority order for success status codes
    success_codes = ["200", "201", "202", "204"]

    response_info = None
    for status_code in success_codes:
        if status_code in route.responses:
            response_info = route.responses[status_code]
            break

    # If no explicit success codes, try any 2xx response
    if response_info is None:
        for status_code, resp_info in route.responses.items():
            if status_code.startswith("2"):
                response_info = resp_info
                break

    if response_info is None or not response_info.content_schema:
        return _DEFAULT_MIME_TYPE

    # If there's only one content type, use it directly
    content_types = list(response_info.content_schema.keys())
    if len(content_types) == 1:
        return content_types[0]

    # When multiple types exist, prefer JSON-compatible types
    json_compatible_types = [
        "application/json",
        "application/vnd.api+json",
        "application/hal+json",
        "application/ld+json",
        "text/json",
    ]
    for ct in json_compatible_types:
        if ct in response_info.content_schema:
            return ct

    # Fall back to the first available content type
    return content_types[0]


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug format.

    Only contains lowercase letters, uppercase letters, numbers, and underscores.
    """
    if not text:
        return ""

    # Replace spaces and common separators with underscores
    slug = re.sub(r"[\s\-\.]+", "_", text)

    # Remove non-alphanumeric characters except underscores
    slug = re.sub(r"[^a-zA-Z0-9_]", "", slug)

    # Remove multiple consecutive underscores
    slug = re.sub(r"_+", "_", slug)

    # Remove leading/trailing underscores
    slug = slug.strip("_")

    return slug


class OpenAPITool(Tool):
    """Tool implementation for OpenAPI endpoints."""

    task_config: TaskConfig = TaskConfig(mode="forbidden")

    def __init__(
        self,
        client: httpx2.AsyncClient,
        route: HTTPRoute,
        director: RequestDirector,
        name: str,
        description: str,
        parameters: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        tags: set[str] | None = None,
        annotations: ToolAnnotations | None = None,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            output_schema=output_schema,
            tags=tags or set(),
            annotations=annotations,
        )
        self._client = client
        self._route = route
        self._director = director

    def __repr__(self) -> str:
        return f"OpenAPITool(name={self.name!r}, method={self._route.method}, path={self._route.path})"

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the HTTP request using RequestDirector."""
        # Build the request — errors here are programming/schema issues,
        # not HTTP failures, so we catch them separately.
        try:
            base_url = str(self._client.base_url) or "http://localhost"
            directed_request = self._director.build(self._route, arguments, base_url)

            # Rebuild through the configured client so its default headers are
            # merged with the directed headers taking priority.
            request = self._client.build_request(
                method=directed_request.method,
                url=str(directed_request.url.copy_with(query=None)),
                params=list(directed_request.url.params.multi_items()),
                headers=list(directed_request.headers.raw),
                # read() materializes streaming bodies (multipart files=)
                # that .content would refuse with RequestNotRead; idempotent
                # for plain byte bodies.
                content=directed_request.read(),
            )

            mcp_headers = get_http_headers()
            if mcp_headers:
                for key, value in mcp_headers.items():
                    if key not in request.headers:
                        request.headers[key] = value
        except Exception as e:
            raise ValueError(
                f"Error building request for {self._route.method.upper()} "
                f"{self._route.path}: {type(e).__name__}: {e}"
            ) from e

        # Send the request and process the response.
        try:
            logger.debug(
                f"run - sending request; headers: {_redact_headers(request.headers)}"
            )

            response = await _send_request(self._client, request)
            _raise_for_status(response)
            return self._build_result(response)

        except httpx2.TimeoutException as exc:
            raise ValueError(f"HTTP request timed out ({type(exc).__name__})") from exc

        except httpx2.RequestError as exc:
            raise ValueError(f"Request error ({type(exc).__name__}): {exc!s}") from exc

    def _build_result(self, response: httpx2.Response) -> ToolResult:
        """Convert a successful response into a tool result.

        JSON bodies become structured content. Other bodies are returned as text
        when they are text, and as base64 image, audio, or blob content otherwise.
        """
        wrap_result = bool(
            self.output_schema and self.output_schema.get("x-fastmcp-wrap-result")
        )
        try:
            result = response.json()
        except ValueError:
            # Not JSON: JSONDecodeError, or UnicodeDecodeError for binary bodies.
            pass
        else:
            return ToolResult(structured_content=self._structure_json(result))

        media_type = _response_media_type(response)
        text = _decode_text_body(response, media_type)
        if text is not None:
            return ToolResult(
                content=text,
                structured_content={"result": text} if wrap_result else None,
            )

        data = base64.b64encode(response.content).decode("ascii")
        mime_type = media_type or "application/octet-stream"
        return ToolResult(
            content=[_binary_content(response, data, mime_type)],
            structured_content={"result": data} if wrap_result else None,
        )

    def _structure_json(self, result: Any) -> dict[str, Any]:
        # Handle structured content based on output schema
        if self.output_schema is not None:
            if self.output_schema.get("x-fastmcp-wrap-result"):
                structured_output = {"result": result}
            else:
                structured_output = result
        elif not isinstance(result, dict):
            structured_output = {"result": result}
        else:
            structured_output = result

        # Structured content must be a dict for the MCP protocol.
        # Wrap non-dict values that slipped through (e.g. a backend
        # returning an array when the schema declared an object).
        if not isinstance(structured_output, dict):
            structured_output = {"result": structured_output}
        return structured_output


# Media types that say nothing about the body; such bodies are checked for text.
_UNTYPED_MEDIA_TYPES = frozenset({"", "application/octet-stream"})

_TEXT_MEDIA_TYPES = frozenset(
    {
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/javascript",
        "application/x-ndjson",
    }
)


def _response_media_type(response: httpx2.Response) -> str:
    return response.headers.get("content-type", "").split(";")[0].strip().lower()


def _decode_text_body(response: httpx2.Response, media_type: str) -> str | None:
    """Return the body as text when it is text, or None when it is binary."""
    if (
        media_type.startswith("text/")
        or media_type in _TEXT_MEDIA_TYPES
        or media_type.endswith(("+xml", "+yaml"))
    ):
        try:
            return response.content.decode(response.encoding or "utf-8")
        except (UnicodeDecodeError, LookupError):
            return None
    if media_type in _UNTYPED_MEDIA_TYPES:
        # Files are often served untyped; strict UTF-8 without NUL bytes is text.
        try:
            text = response.content.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return None if "\x00" in text else text
    return None


def _binary_content(
    response: httpx2.Response, data: str, mime_type: str
) -> ImageContent | AudioContent | EmbeddedResource:
    if mime_type.startswith("image/"):
        return ImageContent(type="image", data=data, mime_type=mime_type)
    if mime_type.startswith("audio/"):
        return AudioContent(type="audio", data=data, mime_type=mime_type)
    # The query string is dropped: it may carry values that should not be echoed.
    uri = str(response.request.url.copy_with(query=None, fragment=None))
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(uri=uri, mime_type=mime_type, blob=data),
    )


class OpenAPIResource(Resource):
    """Resource implementation for OpenAPI endpoints."""

    task_config: TaskConfig = TaskConfig(mode="forbidden")

    def __init__(
        self,
        client: httpx2.AsyncClient,
        route: HTTPRoute,
        director: RequestDirector,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "application/json",
        tags: set[str] | None = None,
        arguments: dict[str, Any] | None = None,
    ):
        super().__init__(
            uri=AnyUrl(uri),
            name=name,
            description=description,
            mime_type=mime_type,
            tags=tags or set(),
        )
        self._client = client
        self._route = route
        self._director = director
        self._arguments = dict(arguments or {})

    def __repr__(self) -> str:
        return f"OpenAPIResource(name={self.name!r}, uri={self.uri!r}, path={self._route.path})"

    async def read(self) -> ResourceResult:
        """Fetch the resource data by making an HTTP request."""
        try:
            base_url = str(self._client.base_url) or "http://localhost"
            directed_request = self._director.build(
                self._route, self._arguments, base_url
            )
            # Build through the configured client so its defaults are applied.
            request = self._client.build_request(
                method=directed_request.method,
                url=str(directed_request.url.copy_with(query=None)),
                params=list(directed_request.url.params.multi_items()),
                headers=list(directed_request.headers.raw),
                # read() materializes streaming bodies (multipart files=)
                # that .content would refuse with RequestNotRead; idempotent
                # for plain byte bodies.
                content=directed_request.read(),
            )
            mcp_headers = get_http_headers()
            if mcp_headers:
                for key, value in mcp_headers.items():
                    if key not in request.headers:
                        request.headers[key] = value

            response = await _send_request(self._client, request)
            _raise_for_status(response)

            content_type = response.headers.get("content-type", "").lower()
            media_type = content_type.split(";")[0].strip()

            # RFC 6839: a "+json" suffix marks a JSON media type
            if media_type == "application/json" or media_type.endswith("+json"):
                result = response.json()
                return ResourceResult(
                    contents=[
                        ResourceContent(
                            content=json.dumps(result), mime_type=media_type
                        )
                    ]
                )
            elif any(ct in content_type for ct in ["text/", "application/xml"]):
                return ResourceResult(
                    contents=[
                        ResourceContent(content=response.text, mime_type=self.mime_type)
                    ]
                )
            else:
                return ResourceResult(
                    contents=[
                        ResourceContent(
                            content=response.content, mime_type=self.mime_type
                        )
                    ]
                )

        except httpx2.TimeoutException as exc:
            raise ValueError(f"HTTP request timed out ({type(exc).__name__})") from exc

        except httpx2.RequestError as exc:
            raise ValueError(f"Request error ({type(exc).__name__}): {exc!s}") from exc


def _path_argument_name(route: HTTPRoute, parameter_name: str) -> str:
    for argument_name, mapping in route.parameter_map.items():
        if mapping["location"] == "path" and mapping["openapi_name"] == parameter_name:
            return argument_name
    return parameter_name


class OpenAPIResourceTemplate(ResourceTemplate):
    """Resource template implementation for OpenAPI endpoints."""

    task_config: TaskConfig = TaskConfig(mode="forbidden")

    def __init__(
        self,
        client: httpx2.AsyncClient,
        route: HTTPRoute,
        director: RequestDirector,
        uri_template: str,
        name: str,
        description: str,
        parameters: dict[str, Any],
        tags: set[str] | None = None,
        mime_type: str = _DEFAULT_MIME_TYPE,
    ):
        super().__init__(
            uri_template=uri_template,
            name=name,
            description=description,
            parameters=parameters,
            tags=tags or set(),
            mime_type=mime_type,
        )
        self._client = client
        self._route = route
        self._director = director

    def __repr__(self) -> str:
        return f"OpenAPIResourceTemplate(name={self.name!r}, uri_template={self.uri_template!r}, path={self._route.path})"

    async def create_resource(
        self,
        uri: str,
        params: dict[str, Any],
        context: Context | None = None,
    ) -> Resource:
        """Create a resource with the given parameters."""
        uri_parts = [f"{key}={value}" for key, value in params.items()]
        arguments = {}
        for parameter in self._route.parameters:
            if parameter.location != "path":
                continue
            argument_name = _path_argument_name(self._route, parameter.name)
            if parameter.name in params:
                arguments[argument_name] = params[parameter.name]
                continue
            normalized_name = parameter.name.replace("-", "_")
            if normalized_name in params:
                arguments[argument_name] = params[normalized_name]

        return OpenAPIResource(
            client=self._client,
            route=self._route,
            director=self._director,
            uri=uri,
            name=f"{self.name}-{'-'.join(uri_parts)}",
            description=self.description or f"Resource for {self._route.path}",
            mime_type=self.mime_type,
            tags=set(self._route.tags or []),
            arguments=arguments,
        )
