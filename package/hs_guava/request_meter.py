"""Count HTTP request attempts, including the native SDK's transport retries."""
from copy import deepcopy
import time


class RequestMeter:
    def __init__(self):
        self.attempts = []

    def reset(self):
        self.attempts = []

    def request(self, request):
        if request.method != 'POST' or not request.url.path.endswith('/chat/completions'):
            return
        index = len(self.attempts)
        request.extensions['guava_attempt'] = index
        self.attempts.append({'started': time.perf_counter(), 'http_status': None, 'usage': None})

    def response(self, response):
        index = response.request.extensions.get('guava_attempt')
        if index is None:
            return
        record = self.attempts[index]
        response.read()
        record['http_status'] = response.status_code
        record['seconds'] = time.perf_counter()-record['started']
        try:
            record['usage'] = response.json().get('usage')
        except (ValueError, AttributeError):
            pass

    def snapshot(self):
        return [{k: v for k, v in row.items() if k != 'started'} for row in deepcopy(self.attempts)]
