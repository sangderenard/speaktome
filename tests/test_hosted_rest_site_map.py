"""The hosted MCP REST hop only forwards API calls and declared map page GETs."""
from unittest.mock import patch

import pytest

from speaktome_mcp.loopback_rest import BearerRest, PreviewRest
from speaktome_mcp.rest import Rest, RestError


@pytest.mark.parametrize("client_type", [BearerRest, PreviewRest])
def test_hosted_rest_allows_declared_map_pages(client_type):
    client = client_type.__new__(client_type)
    client.token = "test-token"
    client.origin = "remote"
    client.tier = "play"
    with patch.object(Rest, "_http", return_value="ok") as http:
        assert client._http("/site/manual/") == "ok"
        assert client._http("/site/manual/speaktome_1-2d3c3fd696.html", method="GET") == "ok"
        assert client._http("/api/whoami") == "ok"
    assert [next(arg for arg in call.args if isinstance(arg, str) and arg.startswith("/"))
            for call in http.call_args_list] == [
        "/site/manual/", "/site/manual/speaktome_1-2d3c3fd696.html", "/api/whoami"
    ]


@pytest.mark.parametrize("client_type", [BearerRest, PreviewRest])
@pytest.mark.parametrize("path,body,method", [
    ("/site/manual/", None, "POST"),
    ("/site/manual/", {}, None),
    ("/site/manual/../private", None, "GET"),
    ("/site/manual/secret.json", None, "GET"),
    ("/config", None, "GET"),
])
def test_hosted_rest_rejects_other_paths_and_site_writes(client_type, path, body, method):
    client = client_type.__new__(client_type)
    client.token = "test-token"
    client.origin = "remote"
    client.tier = "play"
    with patch.object(Rest, "_http") as http:
        with pytest.raises(RestError, match="only the host API may be called"):
            client._http(path, body=body, method=method)
    http.assert_not_called()
