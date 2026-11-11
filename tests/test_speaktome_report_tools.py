"""The existing vocabulary offers only the declared, bounded report action."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaktome_mcp.server import Server, Tools, ToolFail


class Rest:
    def __init__(self): self.calls = []
    def call(self, path, body=None, **kwargs):
        self.calls.append((path, body))
        return {'id': 'a' * 32, 'created': True}


class ReportToolTests(unittest.TestCase):
    def call(self, server, name, arguments):
        result = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                                'params': {'name': name, 'arguments': arguments}})['result']
        return json.loads(result['content'][0]['text']), result['isError']

    def payload(self):
        return {'request_id': 'stable-request', 'report': {'category': 'AUDIT', 'title': 'Scope', 'body': 'Observed.',
                'prompt_history': ['Exact prompt.'], 'next_steps': []}}

    def test_make_uses_existing_rest_and_stable_caller_request(self):
        rest = Rest(); server = Server(rest)
        for _ in range(2):
            result, error = self.call(server, 'make', self.payload())
            self.assertFalse(error)
            self.assertEqual(result['id'], 'a' * 32)
        self.assertEqual(rest.calls[0], rest.calls[1])
        self.assertEqual(rest.calls[0][0], '/api/reports')
        self.assertNotIn('owner_id', rest.calls[0][1])

    def test_unknown_fields_and_invalid_arrays_never_reach_rest(self):
        rest = Rest(); server = Server(rest)
        for field, value in [('owner_id', 1), ('path', '/host/file'), ('prompt_history', []),
                             ('next_steps', 'not-list'), ('prompt_history', [5])]:
            args = self.payload(); args['report'][field] = value
            _, error = self.call(server, 'make', args)
            self.assertTrue(error)
        self.assertEqual(rest.calls, [])

    def test_report_scope_schema_and_forwarding(self):
        server=Server(Rest())
        self.assertEqual(server.table['enter'][1]['properties']['scope']['enum'],['mine','shared','public','all'])
        class StopRest(Rest):
            def call(self,path,body=None,**kwargs):
                self.calls.append((path,body))
                raise RuntimeError('stop after first read')
        for scope in ('mine','shared','public','all'):
            rest=StopRest()
            with self.assertRaises(RuntimeError): Tools(rest).enter(dict(map='reports',scope=scope))
            self.assertEqual(rest.calls[0][0],'/api/reports/map?scope='+scope)
        with self.assertRaises(ToolFail): Tools(Rest()).enter(dict(map='manual',scope='all'))

    def test_report_read_path_and_argument_validation(self):
        tools = Tools(Rest())
        for args in ({'report': '../x'}, {'report': 'a' * 32, 'at': 'elsewhere'}):
            with self.assertRaises(ToolFail): tools.examine(args)
        with self.assertRaises(ToolFail): tools.enter({'map': 'manual', 'offset': 1})
        with self.assertRaises(ToolFail): tools.enter({'map': 'reports', 'instance': 'other'})


if __name__ == '__main__': unittest.main()
