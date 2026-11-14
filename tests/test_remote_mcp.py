"""Offline integration with the real SDK and httpx MockTransport, no provider calls."""
import json
import unittest
from urllib.parse import parse_qs, urlsplit
try:
    import httpx
    from mcp.shared.auth import OAuthToken
except ImportError:
    raise unittest.SkipTest("optional remote-mcp dependency group is not installed")
from speaktome_api.remote_mcp import Provider, RemoteMCP


class MemoryVault:
    def __init__(self):
        self.tokens = self.client = None
    async def clear(self): self.tokens = self.client = None
    async def get_tokens(self): return self.tokens
    async def set_tokens(self, value): self.tokens = value
    async def get_client_info(self): return self.client
    async def set_client_info(self, value): self.client = value


class RemoteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vaults, self.requests, self.handoff = {}, [], None
        self.schema = {'type':'object', '$defs':{'Track':{'type':'string'}},
                       'properties':{'track':{'$ref':'#/$defs/Track'}}, 'required':['track']}
        self.provider = Provider('https://fixture.invalid/mcp', 'https://site.invalid/callback',
                                 frozenset({'https://fixture.invalid'}))
        self.client = RemoteMCP({'music':self.provider}, self.vault,
                               lambda _: httpx.MockTransport(self.respond))
    def vault(self, owner, provider):
        return self.vaults.setdefault((owner, provider), MemoryVault())
    def respond(self, r):
        self.requests.append(r)
        path = r.url.path
        if path.startswith('/.well-known/oauth-protected-resource'):
            return httpx.Response(200,json={'resource':'https://fixture.invalid/mcp',
                  'authorization_servers':['https://fixture.invalid']})
        if path.startswith('/.well-known/'):
            return httpx.Response(200,json={'issuer':'https://fixture.invalid',
                'authorization_endpoint':'https://fixture.invalid/authorize',
                'token_endpoint':'https://fixture.invalid/token',
                'registration_endpoint':'https://fixture.invalid/register',
                'response_types_supported':['code'],'code_challenge_methods_supported':['S256']})
        if path == '/register':
            return httpx.Response(201,json={'client_id':'fixture-client',
                'redirect_uris':['https://site.invalid/callback'], 'token_endpoint_auth_method':'none'})
        if path == '/token':
            form = parse_qs(r.content.decode())
            self.assertEqual(form['code'], ['fixture-code'])
            self.assertTrue(form['code_verifier'])
            return httpx.Response(200,json={'access_token':'fixture-token','token_type':'Bearer','expires_in':3600})
        if r.headers.get('Authorization') != 'Bearer fixture-token':
            return httpx.Response(401,headers={'WWW-Authenticate':'Bearer resource_metadata="https://fixture.invalid/.well-known/oauth-protected-resource/mcp"'})
        if r.method == 'DELETE': return httpx.Response(200)
        if r.method == 'GET': return httpx.Response(405)
        body = json.loads(r.content)
        if 'id' not in body: return httpx.Response(202)
        method = body['method']
        if method == 'initialize':
            result = {'protocolVersion':'2025-11-25', 'serverInfo':{'name':'fixture','version':'1'},'capabilities':{'tools':{}}}
        elif method == 'tools/list':
            cursor = body.get('params',{}).get('cursor')
            result = {'tools':[{'name':'play' if not cursor else 'stop','inputSchema':self.schema}]}
            if not cursor: result['nextCursor']='second'
        elif method == 'tools/call': result = {'content':[{'type':'text','text':'fixture-result'}]}
        else: raise AssertionError(method)
        return httpx.Response(200,json={'jsonrpc':'2.0','id':body['id'],'result':result})
    async def redirect(self, url): self.handoff=url
    async def callback(self):
        params=parse_qs(urlsplit(self.handoff).query)
        self.assertEqual(params['code_challenge_method'],['S256'])
        return 'fixture-code', params['state'][0]
    async def test_real_sdk_authorization_and_lossless_paginated_menu(self):
        menu=await self.client.tools('alice','music',redirect=self.redirect,callback=self.callback)
        self.assertEqual([t['name'] for t in menu], ['play','stop'])
        self.assertEqual(menu[0]['inputSchema'],self.schema)
        self.assertIsNotNone(self.vault('alice','music').tokens)
        result=await self.client.call('alice','music','play',{'track':'x'})
        self.assertEqual(result['content'][0]['text'],'fixture-result')
    async def test_different_owner_cannot_reuse_token(self):
        self.vault('alice','music').tokens=OAuthToken(access_token='fixture-token',token_type='Bearer')
        with self.assertRaises(PermissionError):
            await self.client.tools('bob','music')
        self.assertIsNone(self.vault('bob','music').tokens)
        self.assertFalse(any(r.headers.get('Authorization') for r in self.requests))
    async def test_unlisted_tool_never_called(self):
        self.vault('alice','music').tokens=OAuthToken(access_token='fixture-token',token_type='Bearer')
        with self.assertRaises(BaseExceptionGroup):
            await self.client.call('alice','music','invented',{})
        self.assertFalse(any(b'"tools/call"' in r.content for r in self.requests))
    async def test_unapproved_discovery_origin_is_blocked(self):
        from speaktome_api.remote_mcp import OriginPolicyTransport
        async with httpx.AsyncClient(transport=OriginPolicyTransport(httpx.MockTransport(self.respond),
                                          self.provider.allowed_origins)) as http:
            with self.assertRaises(PermissionError): await http.get('https://other.invalid/token')
        self.assertEqual(self.requests,[])
    def test_configuration_rejects_credentials_and_unlisted_origin(self):
        with self.assertRaises(ValueError):
            Provider('https://user:pass@fixture.invalid/mcp','https://site.invalid/callback',frozenset())



class LifecycleTests(unittest.TestCase):
    def setUp(self):
        from speaktome_api.remote_mcp import ConnectionService
        self.fixture = RemoteTests()
        self.fixture.setUp()
        self.service = ConnectionService({'music':self.fixture.provider}, self.fixture.vault,
                      lambda _:httpx.MockTransport(self.fixture.respond),authorization_timeout=2)
    def tearDown(self): self.service.close()
    def test_handoff_wrong_owner_state_replay_and_disconnect(self):
        import time
        started = self.service.begin('alice','music')
        self.assertEqual(started['status'],'authorization_required')
        state = parse_qs(urlsplit(started['authorization_url']).query)['state'][0]
        with self.assertRaises(PermissionError): self.service.callback('bob','music','fixture-code',state)
        with self.assertRaises(PermissionError): self.service.callback('alice','music','fixture-code','wrong')
        self.service.callback('alice','music','fixture-code',state)
        with self.assertRaises(PermissionError): self.service.callback('alice','music','fixture-code',state)
        for _ in range(100):
            status=self.service.status('alice','music')
            if status['status']=='connected': break
            time.sleep(.01)
        self.assertEqual(status, {'status':'connected','configured':True})
        self.assertEqual(len(self.service.tools('alice','music')),2)
        self.service.disconnect('alice','music')
        self.assertEqual(self.service.status('alice','music'), {'status':'disconnected','configured':False})
    def test_duplicate_begin_reuses_handoff_and_cancel_clears(self):
        a=self.service.begin('alice','music')
        b=self.service.begin('alice','music')
        self.assertEqual(a,b)
        self.service.disconnect('alice','music')
        state=parse_qs(urlsplit(a['authorization_url']).query)['state'][0]
        with self.assertRaises(PermissionError): self.service.callback('alice','music','fixture-code',state)
    def test_timeout_removes_handoff(self):
        import time
        self.service._auth_timeout=.05
        self.service.begin('alice','music')
        time.sleep(.1)
        status=self.service.status('alice','music')
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['error'],'authorization_expired')
        self.assertNotIn('authorization_url',status)

if __name__ == '__main__': unittest.main()

class NetworkPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_pinned_public_ip_and_mixed_private_refusal(self):
        from speaktome_api.remote_mcp import PublicNetworkBackend
        class Backend:
            calls = []
            async def connect_tcp(self, host, port, **kwargs):
                self.calls.append((host,port)); return 'stream'
        backend=Backend()
        async def public(host,port): return ['8.8.8.8']
        net=PublicNetworkBackend(backend,public)
        self.assertEqual(await net.connect_tcp('fixture.invalid',443),'stream')
        self.assertEqual(backend.calls,[('8.8.8.8',443)])
        for address in ['127.0.0.1','10.0.0.1','169.254.169.254','::1','::ffff:127.0.0.1','224.0.0.1']:
            async def mixed(host,port): return ['8.8.8.8',address]
            with self.assertRaises(PermissionError):
                await PublicNetworkBackend(backend,mixed).connect_tcp('fixture.invalid',443)
        self.assertEqual(len(backend.calls),1)


class FailureDiagnosticTests(unittest.TestCase):
    def test_nested_safe_categories_do_not_leak_secrets(self):
        from speaktome_api.remote_mcp import failure_category, ProviderOriginDenied
        import ssl
        secret = 'https://private.invalid/callback?code=secret&state=secret'
        self.assertEqual(failure_category(ExceptionGroup(secret,[ProviderOriginDenied(secret)])),
                         'provider_origin_not_approved')
        error = httpx.ConnectError(secret)
        error.__cause__ = ssl.SSLError(secret)
        self.assertEqual(failure_category(error), 'provider_tls_failed')
        self.assertEqual(failure_category(ValueError(secret)), 'provider_connection_failed')
