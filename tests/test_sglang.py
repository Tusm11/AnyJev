"""SGLang native API contract tests without a SGLang install, a GPU or transformers.

The stub answers with the response shape SGLang 0.5.10 returns for a ``/generate`` call with
``token_ids_logprob`` and ``max_new_tokens=0``: the scores sit under ``meta_info`` as
``output_token_ids_logprobs[0]``, entries are ``[logprob, token_id, text]``, and
``input_token_ids_logprobs`` is null at the prompt's last position.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from anyjev.backends.fake import FakeTokenizer

REAL_SHAPE = {
    "text": "",
    "meta_info": {
        "input_token_ids_logprobs": [None],
        # deliberately not in the order the labels were requested: they are read by id
        "output_token_ids_logprobs": [[[-1.5, 42, " B"], [-0.25, 41, " A"], [-9.0, 43, " C"]]],
    },
}


class _Stub(BaseHTTPRequestHandler):
    request_body = None
    response = REAL_SHAPE

    def do_POST(self):  # noqa: N802 - http.server's name
        _Stub.request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps(_Stub.response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def stub():
    _Stub.request_body, _Stub.response = None, REAL_SHAPE
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _backend(url):
    """A backend with the offline fake tokenizer, so nothing is downloaded and transformers is not
    needed. `FakeTokenizer.encode` maps any text that is not a label to the ids [1, 2]."""
    from anyjev.backends.sglang import SGLangBackend

    be = SGLangBackend.__new__(SGLangBackend)
    be.base_url, be.name, be.api_key = url, "served-model", "EMPTY"
    be.workers, be.timeout, be.source = 1, 10.0, "local-tokenizer"
    be.tokenizer = FakeTokenizer()
    return be


def test_scores_the_prompt_ids_alone_and_maps_the_reply_back_by_id(stub):
    got = _backend(stub).next_token_logprobs(["prompt"], [[41, 42]])

    # no placeholder token appended, no logprob_start_len: the labels are scored at the position
    # after the prompt, and nothing is sampled
    assert _Stub.request_body == {
        "input_ids": [1, 2],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
        "return_logprob": True,
        "token_ids_logprob": [41, 42],
    }
    # the server listed B before A and added C; the result follows the requested order
    np.testing.assert_allclose(got, [[-0.25, -1.5]])
    assert got[0].dtype == np.float64


def test_a_label_the_server_did_not_score_is_an_error(stub):
    with pytest.raises(RuntimeError, match="every requested token id"):
        _backend(stub).next_token_logprobs(["prompt"], [[41, 42, 44]])


def test_a_reply_without_output_logprobs_is_an_error(stub):
    # what the server returns when the scores are not where the backend looks
    _Stub.response = {"meta_info": {"input_token_ids_logprobs": [None],
                                    "output_token_ids_logprobs": [None]}}
    with pytest.raises(RuntimeError, match="did not return token log-probabilities"):
        _backend(stub).next_token_logprobs(["prompt"], [[41]])
    _Stub.response = {"text": ""}
    with pytest.raises(RuntimeError, match="did not return token log-probabilities"):
        _backend(stub).next_token_logprobs(["prompt"], [[41]])


def test_a_null_logprob_for_a_requested_label_is_an_error(stub):
    _Stub.response = {"meta_info": {"output_token_ids_logprobs": [[[None, 41, " A"]]]}}
    with pytest.raises(RuntimeError, match="missing log-probability"):
        _backend(stub).next_token_logprobs(["prompt"], [[41]])


def test_batches_keep_their_order(stub):
    got = _backend(stub).next_token_logprobs(["a", "b"], [[41], [42, 41]])
    np.testing.assert_allclose(got[0], [-0.25])
    np.testing.assert_allclose(got[1], [-1.5, -0.25])


def test_the_tokenizer_comes_from_the_source_not_the_served_alias(monkeypatch):
    """`model` may be a served alias; the tokenizer is loaded from `tokenizer_name`."""
    seen = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            seen.append(name)
            return FakeTokenizer()

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _Auto
    monkeypatch.setitem(sys.modules, "transformers", fake)

    from anyjev.backends.sglang import SGLangBackend

    be = SGLangBackend("http://127.0.0.1:1/", "served-alias", tokenizer_name="some/real-model")
    assert (be.name, be.source, be.base_url) == ("served-alias", "some/real-model", "http://127.0.0.1:1")
    assert seen == ["some/real-model"]


@pytest.mark.engine
def test_real_sglang_server_returns_requested_logprobs():
    base_url = os.environ.get("SGLANG_BASE_URL")
    if not base_url:
        pytest.skip("set SGLANG_BASE_URL to run the SGLang engine smoke test")
    model = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    from anyjev.backends.sglang import SGLangBackend

    backend = SGLangBackend(base_url, model)
    ids = [backend.tokenizer.encode(" A", add_special_tokens=False)[-1],
           backend.tokenizer.encode(" B", add_special_tokens=False)[-1]]
    got = backend.next_token_logprobs(["Answer with one letter:"], [ids])[0]
    assert got.shape == (2,) and np.all(np.isfinite(got))
