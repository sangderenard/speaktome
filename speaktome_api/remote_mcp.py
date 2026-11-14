"""Optional outgoing MCP adapter; host owns identity, secrets and browser handoff.

This is not the standard-library incoming stdio adapter. Install the optional
remote-mcp dependency group through repository setup before importing it.
No provider endpoints, scopes or authorization-server metadata are invented.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata


class ProviderOriginDenied(PermissionError):
    pass


class ProviderAddressDenied(PermissionError):
    pass


def failure_category(exc):
    """Finite safe diagnostic vocabulary; never serialize exception text/URLs."""
    import socket
    import ssl
    import httpcore
    from mcp.client.auth.exceptions import OAuthRegistrationError, OAuthTokenError, OAuthFlowError
    pending, seen, codes = [exc], set(), []
    while pending and len(seen) < 64:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(getattr(current, 'exceptions', ()))
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        if isinstance(current, ProviderOriginDenied): codes.append('provider_origin_not_approved')
        elif isinstance(current, ProviderAddressDenied): codes.append('provider_address_not_public')
        elif isinstance(current, ssl.SSLError): codes.append('provider_tls_failed')
        elif isinstance(current, socket.gaierror): codes.append('provider_dns_failed')
        elif isinstance(current, OAuthRegistrationError): codes.append('provider_registration_failed')
        elif isinstance(current, OAuthTokenError): codes.append('provider_token_failed')
        elif isinstance(current, OAuthFlowError): codes.append('provider_oauth_failed')
        elif isinstance(current, httpx.HTTPStatusError): codes.append('provider_http_rejected')
        elif isinstance(current, (httpx.TimeoutException, httpcore.TimeoutException, TimeoutError)):
            codes.append('provider_network_timeout')
        elif isinstance(current, (httpx.ConnectError, httpcore.ConnectError)):
            codes.append('provider_connect_failed')
    for code in ('provider_origin_not_approved', 'provider_address_not_public',
                 'provider_tls_failed', 'provider_dns_failed', 'provider_registration_failed',
                 'provider_token_failed', 'provider_oauth_failed', 'provider_http_rejected',
                 'provider_network_timeout', 'provider_connect_failed'):
        if code in codes: return code
    return 'provider_connection_failed'


def _https(url: str) -> str:
    p = urlsplit(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password or p.fragment:
        raise ValueError('a credential-free HTTPS URL is required')
    return url


@dataclass(frozen=True)
class Provider:
    """Operator configuration, never taken from arbitrary request parameters."""
    server_url: str
    callback_url: str
    allowed_origins: frozenset[str]
    client_name: str = 'SpeakToMe'

    def __post_init__(self):
        _https(self.server_url)
        _https(self.callback_url)
        if _origin(self.server_url) not in self.allowed_origins:
            raise ValueError('server origin is not approved')
        for origin in self.allowed_origins:
            if _origin(_https(origin)) != origin:
                raise ValueError('allowed origins must be exact HTTPS origins')


def _origin(url: str) -> str:
    p = urlsplit(url)
    port = p.port
    return f'https://{p.hostname}' + (f':{port}' if port and port != 443 else '')


class OriginPolicyTransport(httpx.AsyncBaseTransport):
    """Check every SDK discovery/registration/token/resource request.

    The supplied transport must ALSO enforce deployment DNS/IP egress policy.
    Exact origins alone are not a defense against DNS rebinding. Never give this
    client an unrestricted transport when configurations are user-controlled.
    """
    def __init__(self, delegate: httpx.AsyncBaseTransport, allowed: frozenset[str]):
        self.delegate, self.allowed = delegate, allowed

    async def handle_async_request(self, request):
        _https(str(request.url))
        if _origin(str(request.url)) not in self.allowed:
            raise ProviderOriginDenied('provider requested an unapproved origin')
        return await self.delegate.handle_async_request(request)

    async def aclose(self):
        await self.delegate.aclose()


class RemoteMCP:
    """One operation = one owner-bound SDK session; no shared credentials/cache.

    storage_for must bind a vault to (authenticated owner, provider ID), not a
    caller-supplied account name. transport_for must enforce host network policy.
    Handoff callbacks are required only for explicit user-initiated connect.
    Background tool calls may refresh tokens, but cannot launch authorization.
    """
    def __init__(self, providers: dict[str, Provider],
                 storage_for: Callable[[str, str], TokenStorage], transport_for):
        self.providers = dict(providers)
        self.storage_for = storage_for
        self.transport_for = transport_for

    @asynccontextmanager
    async def session(self, owner: str, provider_id: str, *, redirect=None, callback=None):
        if not owner:
            raise PermissionError('authenticated owner required')
        provider = self.providers[provider_id]
        if (redirect is None) != (callback is None):
            raise ValueError('both authorization handoff callbacks are required')

        async def reject_redirect(_url):
            raise PermissionError('explicit connection authorization required')

        async def checked_redirect(url):
            _https(url)
            if _origin(url) not in provider.allowed_origins:
                raise ProviderOriginDenied('authorization origin is not approved')
            await redirect(url)

        storage = self.storage_for(owner, provider_id)
        if redirect is None and await storage.get_tokens() is None:
            raise PermissionError('explicit connection authorization required')
        oauth = OAuthClientProvider(
            server_url=provider.server_url,
            client_metadata=OAuthClientMetadata(
                client_name=provider.client_name,
                redirect_uris=[provider.callback_url],
                grant_types=['authorization_code', 'refresh_token'],
                response_types=['code'], token_endpoint_auth_method='none',
            ),
            storage=storage,
            redirect_handler=checked_redirect if redirect else reject_redirect,
            callback_handler=callback,
        )
        transport = OriginPolicyTransport(self.transport_for(provider_id), provider.allowed_origins)
        async with httpx.AsyncClient(auth=oauth, transport=transport, timeout=30,
                                     follow_redirects=False, trust_env=False) as http:
            async with streamable_http_client(provider.server_url, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield session

    async def tools(self, owner, provider_id, *, redirect=None, callback=None):
        """Fresh complete provider menu, including nested/$ref schemas unchanged."""
        async with self.session(owner, provider_id, redirect=redirect, callback=callback) as session:
            return await self._tools(session)

    @staticmethod
    async def _tools(session):
        tools, seen, cursor = [], set(), None
        for _ in range(100):
            result = await session.list_tools(cursor=cursor)
            tools.extend(t.model_dump(mode='json', by_alias=True, exclude_none=True) for t in result.tools)
            cursor = result.nextCursor
            if cursor is None:
                return tools
            if cursor in seen:
                raise ValueError('provider repeated a tools cursor')
            seen.add(cursor)
        raise ValueError('provider tool catalogue exceeds page limit')

    async def call(self, owner, provider_id, name, arguments):
        """Host must apply its per-call authorization/approval before invoking."""
        async with self.session(owner, provider_id) as session:
            menu = await self._tools(session)
            if name not in {t['name'] for t in menu}:
                raise ValueError('tool is not in the current provider catalogue')
            result = await session.call_tool(name, arguments=arguments)
            return result.model_dump(mode='json', by_alias=True, exclude_none=True)


class JsonVault:
    """SDK vault adapter over host-protected JSON storage, with no path policy.

    Callbacks are synchronous read()->dict|None, write(dict), clear(). The host
    binds them to one owner/provider and supplies atomic writes and restricted
    permissions. Never serialize this object or expose the underlying record.
    """
    def __init__(self, read, write, clear):
        self._read, self._write, self._clear = read, write, clear

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        data = (self._read() or {}).get('tokens')
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens):
        data = self._read() or {}
        data['tokens'] = tokens.model_dump(mode='json', exclude_none=True)
        self._write(data)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        data = (self._read() or {}).get('client')
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client):
        data = self._read() or {}
        data['client'] = client.model_dump(mode='json', exclude_none=True)
        self._write(data)

    async def clear(self):
        self._clear()


class ConnectionService:
    """Synchronous host facade; OAuth waits on a dedicated bounded event loop.

    The host MUST authenticate every call, protect mutations from CSRF, hide
    callback query strings from logs, and close this object on server shutdown.
    No calls accept a replacement server URL or a different credential owner.
    """
    def __init__(self, providers, storage_for, transport_for, *, authorization_timeout=300,
                 operation_timeout=60, capacity=128):
        import asyncio
        import threading
        self.client = RemoteMCP(providers, storage_for, transport_for)
        self._auth_timeout, self._op_timeout = authorization_timeout, operation_timeout
        self._capacity, self._records = capacity, {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name='remote-mcp-connections')
        self._thread.start()
        self._closed = False

    def _run(self, coroutine):
        import asyncio
        if self._closed:
            coroutine.close()
            raise RuntimeError('connection service closed')
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self._op_timeout + 5)
        except TimeoutError:
            future.cancel()
            raise TimeoutError('provider operation timed out') from None

    def _key(self, owner, provider):
        if not owner:
            raise PermissionError('authenticated owner required')
        if provider not in self.client.providers:
            raise KeyError('unknown provider')
        return owner, provider

    async def _record(self, owner, provider):
        import asyncio
        key = self._key(owner, provider)
        if key not in self._records:
            if len(self._records) >= self._capacity:
                # Only discard completed/inactive runtime state, never vaults.
                idle = next((k for k,v in self._records.items()
                             if not v['lock'].locked() and (v['task'] is None or v['task'].done())), None)
                if idle is None:
                    raise RuntimeError('connection capacity reached')
                del self._records[idle]
            self._records[key] = dict(lock=asyncio.Lock(), task=None, status='disconnected',
                                      ready=None, reply=None, url=None, state=None, error=None)
        return self._records[key]

    async def _status(self, owner, provider):
        r = await self._record(owner, provider)
        configured = await self.client.storage_for(owner, provider).get_tokens() is not None
        status = r['status']
        if status == 'disconnected' and configured:
            status = 'authorized'  # Stored authorization, not a live health claim.
        result = {'status':status, 'configured':configured}
        if r['error']: result['error'] = r['error']
        if r['url'] and status == 'authorization_required':
            result['authorization_url'] = r['url']
        return result

    def status(self, owner, provider): return self._run(self._status(owner, provider))

    async def _begin(self, owner, provider):
        import asyncio
        from urllib.parse import parse_qs
        r = await self._record(owner, provider)
        if r['task'] is not None and not r['task'].done():
            return await self._status(owner, provider)
        if r['lock'].locked():
            raise RuntimeError('connection is busy')
        r.update(status='connecting', error=None, url=None, state=None,
                 ready=asyncio.Event(), reply=self._loop.create_future())

        async def redirect(url):
            states = parse_qs(urlsplit(url).query).get('state', [])
            if len(states) != 1 or not states[0]:
                raise ValueError('provider handoff has no unique state')
            r.update(url=url, state=states[0], status='authorization_required')
            r['ready'].set()

        async def callback(): return await r['reply']

        async def connect():
            try:
                async with r['lock']:
                    async with asyncio.timeout(self._auth_timeout):
                        await self.client.tools(owner, provider, redirect=redirect, callback=callback)
                    r['status'] = 'connected'
            except asyncio.CancelledError:
                r['status'] = 'disconnected'
                raise
            except TimeoutError:
                r.update(status='failed', error='authorization_expired')
            except Exception as exc:
                # Raw OAuth exceptions can contain codes/state/tokens. Never return/log them.
                r.update(status='failed', error=failure_category(exc))
            finally:
                r.update(url=None, state=None)
                r['ready'].set()
                if not r['reply'].done(): r['reply'].cancel()

        r['task'] = asyncio.create_task(connect())
        try:
            await asyncio.wait_for(r['ready'].wait(), timeout=min(10, self._op_timeout))
        except TimeoutError:
            pass  # Client polls status while metadata discovery continues.
        return await self._status(owner, provider)

    def begin(self, owner, provider): return self._run(self._begin(owner, provider))

    async def _callback(self, owner, provider, code, state):
        import secrets
        r = await self._record(owner, provider)
        if (r['status'] != 'authorization_required' or not isinstance(state,str)
            or not isinstance(code,str) or not code or len(code)>16384
            or not r['state'] or not secrets.compare_digest(state, r['state'])
            or r['reply'].done()):
            raise PermissionError('invalid or expired authorization callback')
        r.update(status='finishing', url=None, state=None)
        r['reply'].set_result((code, state))
        return {'status':'finishing'}

    def callback(self, owner, provider, code, state):
        return self._run(self._callback(owner, provider, code, state))

    async def _operation(self, owner, provider, name=None, arguments=None):
        import asyncio
        r = await self._record(owner, provider)
        if r['task'] is not None and not r['task'].done():
            raise RuntimeError('connection authorization is in progress')
        async with r['lock']:
            async with asyncio.timeout(self._op_timeout):
                if name is None: return await self.client.tools(owner, provider)
                return await self.client.call(owner, provider, name, arguments)

    def tools(self, owner, provider): return self._run(self._operation(owner, provider))
    def call(self, owner, provider, name, arguments):
        return self._run(self._operation(owner, provider, name, arguments))

    async def _disconnect(self, owner, provider):
        import asyncio
        r = await self._record(owner, provider)
        if r['task'] is not None and not r['task'].done():
            r['task'].cancel()
            await asyncio.gather(r['task'], return_exceptions=True)
        async with r['lock']:
            await self.client.storage_for(owner, provider).clear()
            r.update(status='disconnected', error=None, url=None, state=None)
        return {'status':'disconnected'}

    def disconnect(self, owner, provider): return self._run(self._disconnect(owner, provider))

    def close(self):
        import asyncio
        async def stop():
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._closed: return
        self._run(stop())
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._thread.is_alive(): self._loop.close()


class PublicNetworkBackend:
    """Resolve once, reject nonpublic answers, connect the checked IP directly.

    httpcore supplies the original TLS server name to the returned stream, so
    certificate verification and SNI remain bound to the configured hostname.
    """
    def __init__(self, delegate=None, resolver=None):
        import httpcore
        self.delegate = delegate or httpcore.AnyIOBackend()
        self.resolver = resolver

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        import asyncio
        import ipaddress
        import socket
        async def resolve():
            if self.resolver is not None: return await self.resolver(host, port)
            rows = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
            return list(dict.fromkeys(row[4][0] for row in rows))
        addresses = await asyncio.wait_for(resolve(), timeout=timeout)
        if not addresses:
            raise OSError('provider hostname has no addresses')
        for address in addresses:
            ip = ipaddress.ip_address(address)
            effective = getattr(ip, 'ipv4_mapped', None) or ip
            if not effective.is_global or effective.is_multicast or effective.is_reserved:
                raise ProviderAddressDenied('provider hostname resolved to a nonpublic address')
        # The destination is an IP now; the backend cannot re-resolve the hostname.
        return await self.delegate.connect_tcp(addresses[0], port, timeout=timeout,
                    local_address=local_address, socket_options=socket_options)

    async def connect_unix_socket(self, *args, **kwargs):
        raise PermissionError('Unix sockets are not permitted for provider connections')

    async def sleep(self, seconds):
        import asyncio
        await asyncio.sleep(seconds)


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream): self.stream = stream
    async def __aiter__(self):
        async for chunk in self.stream: yield chunk
    async def aclose(self): await self.stream.aclose()


class PublicHTTPTransport(httpx.AsyncBaseTransport):
    """Verified TLS + public-IP transport for operator-allowlisted providers."""
    def __init__(self):
        import httpcore
        import ssl
        self.pool = httpcore.AsyncConnectionPool(ssl_context=ssl.create_default_context(),
                           network_backend=PublicNetworkBackend(), max_connections=4, retries=0)

    async def handle_async_request(self, request):
        import httpcore
        _https(str(request.url))
        req = httpcore.Request(method=request.method,
            url=httpcore.URL(scheme=request.url.raw_scheme, host=request.url.raw_host,
                             port=request.url.port, target=request.url.raw_path),
            headers=request.headers.raw, content=request.stream, extensions=request.extensions)
        response = await self.pool.handle_async_request(req)
        return httpx.Response(response.status, headers=response.headers,
                              stream=_ResponseStream(response.stream), extensions=response.extensions)

    async def aclose(self): await self.pool.aclose()
