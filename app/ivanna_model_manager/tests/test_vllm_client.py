"""v5.0: тесты VLLMClient — payload-конструкция и разбор ответов.

Сервер не нужен: requests мокается, проверяется логика клиента.
"""
from unittest.mock import patch, MagicMock

from django.test import SimpleTestCase

from ivanna_model_manager.services.vllm_client import VLLMClient


class BuildPayloadTests(SimpleTestCase):
    def test_basic_payload(self):
        p = VLLMClient.build_payload('mymodel', 'Hello', max_tokens=256, temperature=0.5)
        self.assertEqual(p['model'], 'mymodel')
        self.assertEqual(p['prompt'], 'Hello')
        self.assertEqual(p['max_tokens'], 256)
        self.assertEqual(p['temperature'], 0.5)
        self.assertFalse(p['stream'])

    def test_unknown_kwargs_dropped(self):
        # gen_config сессии может нести ключи, которых нет в OpenAI API
        p = VLLMClient.build_payload('m', 'x', max_tokens=10,
                                     quantization='int4', use_rag=True, rag_config={})
        self.assertNotIn('quantization', p)
        self.assertNotIn('use_rag', p)
        self.assertNotIn('rag_config', p)

    def test_vllm_extensions_passed(self):
        p = VLLMClient.build_payload('m', 'x', top_k=40, repetition_penalty=1.1)
        self.assertEqual(p['top_k'], 40)
        self.assertEqual(p['repetition_penalty'], 1.1)

    def test_top_k_zero_not_sent(self):
        p = VLLMClient.build_payload('m', 'x', top_k=0)
        self.assertNotIn('top_k', p)


class GenerateTests(SimpleTestCase):
    def _mock_response(self, status=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = json_data or {}
        return resp

    def test_success_parses_openai_shape(self):
        resp = self._mock_response(json_data={
            'choices': [{'text': ' The answer is 42.'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 7, 'total_tokens': 19},
        })
        with patch('ivanna_model_manager.services.vllm_client.requests.post', return_value=resp) as post:
            result = VLLMClient().generate('mymodel', 'Question?', max_tokens=64)
        self.assertTrue(result['success'])
        self.assertEqual(result['response'], ' The answer is 42.')
        self.assertEqual(result['input_tokens'], 12)
        self.assertEqual(result['output_tokens'], 7)
        self.assertEqual(result['total_tokens'], 19)
        self.assertEqual(result['device'], 'vllm')
        # payload ушёл на /v1/completions
        self.assertIn('/v1/completions', post.call_args[0][0])
        self.assertEqual(post.call_args[1]['json']['model'], 'mymodel')

    def test_http_error_is_dict_error_not_exception(self):
        resp = self._mock_response(status=500, json_data={'error': {'message': 'boom'}})
        with patch('ivanna_model_manager.services.vllm_client.requests.post', return_value=resp):
            result = VLLMClient().generate('m', 'x')
        self.assertFalse(result['success'])
        self.assertIn('HTTP 500', result['error'])

    def test_timeout_is_dict_error(self):
        import requests as requests_lib
        with patch('ivanna_model_manager.services.vllm_client.requests.post',
                   side_effect=requests_lib.exceptions.Timeout('t')):
            result = VLLMClient(timeout=5).generate('m', 'x')
        self.assertFalse(result['success'])
        self.assertIn('timed out', result['error'])

    def test_connection_error_is_dict_error(self):
        import requests as requests_lib
        with patch('ivanna_model_manager.services.vllm_client.requests.post',
                   side_effect=requests_lib.exceptions.ConnectionError('refused')):
            result = VLLMClient().generate('m', 'x')
        self.assertFalse(result['success'])
        self.assertIn('unreachable', result['error'])

    def test_empty_choices_is_error(self):
        resp = self._mock_response(json_data={'choices': []})
        with patch('ivanna_model_manager.services.vllm_client.requests.post', return_value=resp):
            result = VLLMClient().generate('m', 'x')
        self.assertFalse(result['success'])
        self.assertIn('no choices', result['error'])

    def test_auth_header_when_key_set(self):
        c = VLLMClient(api_key='test-key-123')
        h = c._headers()
        self.assertEqual(h['Authorization'], 'Bearer test-key-123')
