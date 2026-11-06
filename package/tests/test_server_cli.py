"""Exercise the actual vla-eval CLI parser without network, GPU or model loading."""
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class ServerCLIChecks(unittest.TestCase):
    def test_original_parser_missing_name_and_malformed_tools_are_metered_decisions(self):
        try:
            from guava.llm.client import _parse_tool_from_text
            from hs_guava.model_server import GuavaPolicyServer
        except ImportError:
            self.skipTest('Install the native Guava source for malformed-output integration')
        from hs_guava.codec import decode

        def compatible(text, mode):
            name, arguments = _parse_tool_from_text(text)
            return 'tool', {'name': name, 'arguments': arguments}

        parser = SimpleNamespace(compatible=compatible, audit=[])
        model, revision = 'frozen-cpu-fixture', 'a'*40
        reply = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=''))],
            usage=SimpleNamespace(model_dump=lambda: {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}))
        fake = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=model)])),
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: reply)))
        context = SimpleNamespace(session_id='test', episode_id='test')
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('os.environ', {'GUAVA_MODEL_REVISION': revision}), \
                 patch('hs_guava.model_server.load_source', return_value=(None,None,parser,None,None,None,None,None)), \
                 patch('hs_guava.model_server.manifest', return_value={'model': {'name': model, 'revision': revision}}), \
                 patch('openai.OpenAI', return_value=fake) as client:
                server = GuavaPolicyServer(audit_dir=folder)
            try:
                cases = [('<tool_call>{"arguments": {}}</tool_call>', 'invalid'),
                         ('<tool_call>{"name": 5, "arguments": {}}</tool_call>', 'invalid'),
                         ('<tool_call>{"name": "move", "arguments": {"target_position": [NaN,0,0]}}</tool_call>', 'invalid'),
                         ('<tool_call>{"name": "release", "arguments": {}}</tool_call>', 'tool')]
                for text, expected in cases:
                    reply.choices[0].message.content = text
                    action = server.predict({'guava_request': {'system_prompt': 'fixture',
                        'message': {'role': 'user', 'content': 'fixture'}}}, context)
                    self.assertEqual(decode(action)['kind'], expected)
                    record = json.loads((Path(folder)/'requests.jsonl').read_text().splitlines()[-1])
                    self.assertEqual(record['text'], text)
                    self.assertEqual(record['usage']['total_tokens'], 10)
                    self.assertEqual(bool(record['parser_error']), expected == 'invalid')
            finally:
                server.log.close()
                client.call_args.kwargs['http_client'].close()

    def test_no_network_server_construction_through_standard_entrypoint(self):
        try:
            from vla_eval.model_servers.serve import run_server
            from hs_guava.model_server import GuavaPolicyServer
        except ImportError:
            self.skipTest('Install the native evaluation environment for CLI integration')
        name, revision = 'frozen-cpu-fixture', 'a'*40
        instance = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=name)])))
        with tempfile.TemporaryDirectory() as folder:
            argv = ['guava-server', '--host', '127.0.0.1', '--port', '8118', '--args.audit_dir', folder]
            with patch('sys.argv', argv), patch.dict('os.environ', {'GUAVA_MODEL_REVISION': revision}), \
                 patch('hs_guava.model_server.load_source', return_value=(None,None,None,None,None,None,None,None)), \
                 patch('hs_guava.model_server.manifest', return_value={'model': {'name': name, 'revision': revision}}), \
                 patch('openai.OpenAI', return_value=instance) as client, \
                 patch('vla_eval.model_servers.serve.serve') as serving:
                run_server(GuavaPolicyServer)
                server = serving.call_args.args[0]
                self.assertEqual(serving.call_args.kwargs['host'], '127.0.0.1')
                self.assertEqual(server.model, name)
                self.assertTrue((Path(folder)/'requests.jsonl').exists())
                server.log.close()
                client.call_args.kwargs['http_client'].close()


class ServerProtocolChecks(unittest.IsolatedAsyncioTestCase):
    async def test_actual_network_handshake_episode_and_semantic_action_transport(self):
        try:
            from vla_eval.api import serve_background
            from harnessstudio.context.policy import PolicyHandle, PolicyProfile
            from hs_guava.model_server import GuavaPolicyServer
        except ImportError:
            self.skipTest('Install pinned evaluation dependencies for protocol integration')
        from hs_guava.codec import ACTION_DIM, decode
        model, revision = 'frozen-cpu-fixture', 'a'*40
        reply = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='Task complete.'))],
                                usage=SimpleNamespace(model_dump=lambda: {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}))
        client_instance = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=model)])),
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: reply)))
        compatible = SimpleNamespace(compatible=lambda content, mode: ('terminal', 'complete'), audit=[])
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('os.environ', {'GUAVA_MODEL_REVISION': revision}), \
                 patch('hs_guava.model_server.load_source', return_value=(None,None,compatible,None,None,None,None,None)), \
                 patch('hs_guava.model_server.manifest', return_value={'model': {'name': model, 'revision': revision}}), \
                 patch('openai.OpenAI', return_value=client_instance) as client:
                server = GuavaPolicyServer(audit_dir=folder)
            try:
                with serve_background(server, host='127.0.0.1', port=0) as serving:
                    policy = PolicyHandle(serving.url, PolicyProfile(execute_horizon=1), timeout=10)
                    try:
                        await policy.open({'task': {'name': 'synthetic-semantic-fixture'}})
                        chunk = await policy.act({'task_description': 'fixture', 'guava_request': {
                            'system_prompt': 'fixture', 'message': {'role': 'user', 'content': 'fixture'}}})
                        self.assertEqual(np.asarray(chunk).shape, (1,ACTION_DIM))
                        self.assertEqual(decode(chunk[0]), {'kind': 'terminal', 'value': 'complete'})
                    finally:
                        await policy.close()
            finally:
                server.log.close()
                client.call_args.kwargs['http_client'].close()
