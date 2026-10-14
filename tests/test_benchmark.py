import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from benchmark import percentile, request_one, summarize


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if payload['model'] == 'bad':
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'choices': [{'message': {'content': 'hello'}}],
                                    'usage': {'completion_tokens': 2}}).encode())
    def log_message(self, *args):
        pass


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_and_failures(self):
        rows = [{'ok': True, 'latency_s': 1, 'completion_tokens': 4},
                {'ok': False, 'latency_s': 9}]
        result = summarize(rows, 2)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['output_tokens_per_s'], 2)
        self.assertEqual(result['latency_s']['p99'], 1)
        self.assertIsNone(percentile([], .99))

    def test_missing_usage_is_unknown(self):
        self.assertIsNone(summarize([{'ok': True, 'latency_s': 1}], 2)['output_tokens_per_s'])

    def test_http_success_and_error(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/v1'
            good = request_one(url, 'test', 'hi', 8, 2)
            self.assertTrue(good['ok'])
            self.assertEqual(good['completion_tokens'], 2)
            self.assertFalse(request_one(url, 'bad', 'hi', 8, 2)['ok'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
