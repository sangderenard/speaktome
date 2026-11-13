"""Shared account-group HTTP contract, separate from report/content storage."""
from http.cookies import SimpleCookie, CookieError
import json
import re
import sqlite3

API_VERSION = 1


def route(self,path,method,*,service_type,accounts,account_cookie,AccessError,SchemaVersionError):
    self._set_cookie=[]
    if method=='POST':
        self.close_connection=True
    if self._is_viewer():
        return self._refuse_viewer()
    try:
        if self._account() is None:
            raise AccessError('sign in to use account groups',401)
        if '?' in self.path:
            raise AccessError('groups accept no query parameters')
        jar=SimpleCookie(self.headers.get('Cookie',''))
        token=jar[account_cookie].value if account_cookie in jar else None
        service=service_type(accounts)
        if method=='GET':
            if path=='/api/account/groups':
                return self._json(service.listing(token))
            if path=='/api/account/groups/invitations':
                return self._json(service.invitations(token))
            match=re.fullmatch(r'/api/account/groups/([0-9a-f]{32})',path)
            if match:
                return self._json(service.examine(token,match[1]))
            raise AccessError('group not found',404)
        if method!='POST':
            raise AccessError('not found',404)
        expected=('https://' if self._is_secure() else 'http://')+self.headers.get('Host','')
        if self.headers.get('Origin') not in (None,expected) or self.headers.get('Sec-Fetch-Site')=='cross-site':
            raise AccessError('cross-origin group writes are not allowed',403)
        if self.headers.get('Content-Type','').split(';',1)[0].strip().lower()!='application/json':
            raise AccessError('application/json required',415)
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1:
            raise AccessError('one Content-Length is required')
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=16384:
            raise AccessError('group request exceeds size limit',413)
        body=json.loads(self.rfile.read(length))
        if path=='/api/account/groups':
            result=service.create(token,body)
            return self._json(result,201 if result['created'] else 200)
        response=re.fullmatch(r'/api/account/groups/invitations/([0-9a-f]{32})',path)
        if response:
            return self._json(service.respond(token,response[1],body))
        match=re.fullmatch(r'/api/account/groups/([0-9a-f]{32})/(invite|members|leave)',path)
        if not match:
            raise AccessError('group not found',404)
        if match[2]=='members' and isinstance(body,dict):
            body.setdefault('role',None)
        method_name={'invite':'invite','members':'member','leave':'leave'}[match[2]]
        return self._json(getattr(service,method_name)(token,match[1],body))
    except AccessError as error:
        return self._json({'error':str(error)},error.status)
    except (ValueError,TypeError,CookieError,RecursionError,UnicodeError):
        return self._json({'error':'invalid group request'},400)
    except (sqlite3.Error,SchemaVersionError):
        return self._json({'error':'group storage unavailable; no success is confirmed'},503)
