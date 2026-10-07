"""SGLang backend through the native ``/generate`` API.

The native endpoint scores arbitrary token ids without sampling: send the prompt's
token ids alone with ``max_new_tokens=0``, ``return_logprob=True`` and
``token_ids_logprob`` set to the labels AnyJev asked for. The server returns the
log-probabilities of those ids at the position after the prompt, under
``meta_info["output_token_ids_logprobs"][0]``, as ``[logprob, token_id, text]``
entries; they are mapped back to the requested ids by id.

    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000
    Decider(SGLangBackend("http://localhost:30000", "Qwen/Qwen3-8B"))

Checked on SGLang 0.5.10 (Qwen2.5-7B-Instruct, bf16) against ``HFBackend`` with
``scripts/sglang_parity.py``. The values are full-vocabulary log-probabilities of
the requested ids, not renormalized over them. This backend needs a SGLang server
and does not expose L2 hidden states.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import urllib.request
from typing import List, Sequence

import numpy as np


class SGLangBackend:
    """Next-token log-probabilities from SGLang's native generation API."""

    def __init__(self, base_url: str, model: str, tokenizer_name: str | None = None,
                 api_key: str = "EMPTY", workers: int = 16, timeout: float = 120.0):
        from transformers import AutoTokenizer

        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        # `model` may be a served alias; the tokenizer has to come from something loadable.
        self.source = tokenizer_name or model
        self.tokenizer = AutoTokenizer.from_pretrained(self.source)

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        body = {
            "input_ids": self.tokenizer.encode(prompt, add_special_tokens=False),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "return_logprob": True,
            "token_ids_logprob": [int(i) for i in ids],
        }
        req = urllib.request.Request(
            self.base_url + "/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            out = json.load(response)

        rows = (out.get("meta_info") or {}).get("output_token_ids_logprobs")
        if not rows or rows[0] is None:
            raise RuntimeError("SGLang did not return token log-probabilities")
        # entries are [logprob, token_id, text]; read them by id, not by position
        got = {int(entry[1]): entry[0] for entry in rows[0]}
        missing = [int(i) for i in ids if int(i) not in got]
        if missing:
            raise RuntimeError(
                "SGLang did not return log-probabilities for every requested token id: "
                f"missing {missing}")
        values = [got[int(i)] for i in ids]
        if any(value is None for value in values):
            raise RuntimeError("SGLang returned a missing log-probability for a requested token id")
        return np.asarray(values, dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        with cf.ThreadPoolExecutor(self.workers) as executor:
            return list(executor.map(self._one, prompts, token_ids))
