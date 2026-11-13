"""Guestbook HTTP contract. The host supplies its Accounts-bound SQL service.

The existing make/examine/navigation vocabulary acts on experience reports.
There is no generic method dispatcher, host-path upload, or second authority.
"""
from http.cookies import SimpleCookie, CookieError
import json
import re
import sqlite3
from urllib.parse import parse_qs

API_VERSION = 2


def route(self, path, method, *, service_type, accounts, account_cookie, ReportError, SchemaVersionError, max_body):
    self._set_cookie = []
    if method == 'POST':
        self.close_connection = True
    if self._is_viewer():
        return self._refuse_viewer()
    try:
        public = method == 'GET' and (path == '/api/reports/public' or re.fullmatch(r'/api/reports/public/[0-9a-f]{32}', path))
        if not public and self._account() is None:
            raise ReportError('sign in to use experience reports', 401)
        jar = SimpleCookie(self.headers.get('Cookie', ''))
        token = jar[account_cookie].value if account_cookie in jar else None
        service = service_type(accounts)
        query = parse_qs(self.path.partition('?')[2], keep_blank_values=True)
        if any(len(v) != 1 for v in query.values()):
            raise ReportError('duplicate query parameters')
        if method == 'GET':
            if public:
                if path == '/api/reports/public':
                    if set(query) - {'offset','category'}:
                        raise ReportError('unexpected public report query')
                    return self._json(service.listing(None, public=True, scope='public',
                        offset=int(query.get('offset',['0'])[0]), category=query.get('category',[None])[0]))
                if query:
                    raise ReportError('unexpected report query')
                return self._json(service.examine(None,path.rsplit('/',1)[1],public=True))
            if path in ('/api/reports', '/api/reports/map'):
                if set(query) - {'offset', 'category', 'scope'}:
                    raise ReportError('unexpected report query')
                args = {'offset': int(query.get('offset', ['0'])[0]), 'category': query.get('category', [None])[0],
                        'scope': query.get('scope', ['mine'])[0]}
                return self._json(service.map(token, **args) if path.endswith('/map') else service.listing(token, **args))
            match = re.fullmatch(r'/api/reports/([0-9a-f]{32})', path)
            if not match:
                raise ReportError('report not found', 404)
            if query:
                raise ReportError('unexpected report query')
            return self._json(service.examine(token, match[1]))
        sharing = re.fullmatch(r'/api/reports/([0-9a-f]{32})/sharing', path)
        if method != 'POST' or (path != '/api/reports' and not sharing):
            raise ReportError('not found', 404)
        if query:
            raise ReportError('report writes accept no query parameters')
        expected = ('https://' if self._is_secure() else 'http://') + self.headers.get('Host', '')
        if self.headers.get('Origin') not in (None, expected) or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise ReportError('cross-origin report writes are not allowed', 403)
        if self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
            raise ReportError('application/json required', 415)
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
            raise ReportError('one Content-Length is required')
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= max_body:
            raise ReportError('report request exceeds size limit', 413)
        body = json.loads(self.rfile.read(length))
        if sharing:
            if isinstance(body,dict):
                body.setdefault('group_id',None)
            return self._json(service.sharing(token,sharing[1],body))
        result = service.make(token, body)
        return self._json(result, 201 if result['created'] else 200)
    except ReportError as error:
        return self._json({'error': str(error)}, error.status)
    except (ValueError, TypeError, CookieError, RecursionError, UnicodeError):
        return self._json({'error': 'invalid report request'}, 400)
    except (sqlite3.Error, SchemaVersionError):
        return self._json({'error': 'report storage unavailable; no success is confirmed'}, 503)
