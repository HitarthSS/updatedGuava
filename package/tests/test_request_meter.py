"""Real SDK retries over an in-memory HTTP transport; no network/model calls."""
import unittest


class RequestMeterChecks(unittest.TestCase):
    def test_retried_post_requests_are_counted_and_usage_is_not_duplicated(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest('Install Guava evaluation dependencies for SDK retry check')
        from hs_guava.request_meter import RequestMeter
        meter = RequestMeter()
        count = 0
        def handle(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(503, json={'error': {'message': 'synthetic overload', 'type': 'server_error'}})
            return httpx.Response(200, json={'id': 'cpu-test', 'object': 'chat.completion', 'created': 0,
                'model': 'frozen-test-fixture', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'ok'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}})
        with httpx.Client(transport=httpx.MockTransport(handle),
                          event_hooks={'request': [meter.request], 'response': [meter.response]}) as http:
            client = OpenAI(base_url='http://127.0.0.1:1/v1', api_key='EMPTY', max_retries=2, http_client=http)
            client._calculate_retry_timeout = lambda *args, **kwargs: 0
            output = client.chat.completions.create(model='frozen-test-fixture', messages=[{'role': 'user', 'content': 'test'}])
        self.assertEqual(count, 2)
        self.assertEqual(output.usage.total_tokens, 12)
        self.assertEqual([r['http_status'] for r in meter.snapshot()], [503, 200])
        self.assertIsNone(meter.snapshot()[0]['usage'])
        self.assertEqual(meter.snapshot()[1]['usage']['total_tokens'], 12)
        meter.reset()
        self.assertEqual(meter.snapshot(), [])
