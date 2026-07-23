# Copyright 2026 Individual Contributor: Michael Lavery
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDPO response-position alignment guarantee (CPU).

SDPO's self-teacher is scored on the SAME response tokens as the student, but its
prompt is feedback-augmented and therefore a DIFFERENT length than the student's
prompt. This test locks the correctness argument that verl's response extraction
(``response_from_nested``) counts the response window back from the *sequence end*,
so student and teacher per-response-token outputs align to the identical response
tokens regardless of prompt length -- and that ``response_to_sequence_nested``, which
maps the teacher's window back onto the student's sequence, is its exact inverse. No GPUs.
"""

import torch

from verl.workers.utils.padding import response_from_nested, response_to_sequence_nested


def _next_token_values(seq_ids: list[int]) -> torch.Tensor:
    """value[t] = the token predicted at position t (= seq_ids[t+1]); sentinel at the end.

    verl left-shifts model output by one for log-probs, so the value stored at position
    ``t`` corresponds to predicting token ``t+1``. Encoding the predicted token id as the
    value lets us assert *which* response tokens the extracted window lines up with.
    """
    vals = [seq_ids[t + 1] for t in range(len(seq_ids) - 1)] + [-1]
    return torch.tensor(vals, dtype=torch.float64)


def _nested(seqs: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(list(seqs), layout=torch.jagged)


def test_response_window_aligns_across_prompt_lengths():
    response = [901, 902, 903, 904]  # identical response tokens for student and teacher
    r = len(response)

    # Student and teacher prompts deliberately differ in length.
    student_prompt = [11, 12, 13]
    teacher_prompt = [21, 22, 23, 24, 25, 26, 27]  # feedback-augmented -> longer

    student_seq = student_prompt + response
    teacher_seq = teacher_prompt + response

    values = _nested([_next_token_values(student_seq), _next_token_values(teacher_seq)])
    # response_mask carries only per-sequence response length via its offsets.
    response_mask = _nested([torch.ones(r), torch.ones(r)])

    extracted = response_from_nested(values, response_mask)

    student_resp = extracted[0]
    teacher_resp = extracted[1]
    expected = torch.tensor(response, dtype=torch.float64)

    # Both extract values that predict exactly the response tokens -> aligned.
    torch.testing.assert_close(student_resp, expected)
    torch.testing.assert_close(teacher_resp, expected)


def test_alignment_holds_for_ragged_batch():
    # Different prompt AND response lengths per sample, student vs teacher prompts differ.
    samples = [
        {"sp": [1, 2], "tp": [7, 7, 7, 7], "resp": [50, 51, 52]},
        {"sp": [3, 3, 3, 3, 3], "tp": [9], "resp": [60, 61]},
    ]
    student_vals, teacher_vals, student_rm, teacher_rm, expected = [], [], [], [], []
    for s in samples:
        student_vals.append(_next_token_values(s["sp"] + s["resp"]))
        teacher_vals.append(_next_token_values(s["tp"] + s["resp"]))
        student_rm.append(torch.ones(len(s["resp"])))
        teacher_rm.append(torch.ones(len(s["resp"])))
        expected.append(torch.tensor(s["resp"], dtype=torch.float64))

    student_ext = response_from_nested(_nested(student_vals), _nested(student_rm))
    teacher_ext = response_from_nested(_nested(teacher_vals), _nested(teacher_rm))

    for i in range(len(samples)):
        torch.testing.assert_close(student_ext[i], expected[i])
        torch.testing.assert_close(teacher_ext[i], expected[i])


def test_response_to_sequence_nested_inverts_response_from_nested():
    """The teacher top-k is scattered onto the student's sequence; round-tripping must be lossless.

    Uses a trailing dense dim (the top-k axis) since that is how SDPO calls it.
    """
    samples = [{"prompt_len": 3, "resp_len": 4}, {"prompt_len": 6, "resp_len": 2}]
    topk = 5
    prompt_lens = torch.tensor([s["prompt_len"] for s in samples])
    response_lens = torch.tensor([s["resp_len"] for s in samples])
    max_resp = int(response_lens.max())

    # Padded (bsz, max_resp, topk) response-aligned values, distinct per position.
    padded = torch.zeros(len(samples), max_resp, topk, dtype=torch.float64)
    for i, s in enumerate(samples):
        for j in range(s["resp_len"]):
            padded[i, j] = torch.arange(topk, dtype=torch.float64) + 100 * i + 10 * j + 1

    sequence = response_to_sequence_nested(padded, prompt_lens, response_lens)

    # Valid lengths are prompt+response, not a fixed width.
    torch.testing.assert_close(sequence.offsets().diff(), (prompt_lens + response_lens).to(torch.int64))

    # response_from_nested must recover exactly what was scattered in.
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(s["resp_len"], dtype=torch.float64) for s in samples], layout=torch.jagged
    )
    recovered = response_from_nested(sequence, response_mask)
    for i, s in enumerate(samples):
        torch.testing.assert_close(recovered[i], padded[i, : s["resp_len"]])

    # Everything outside the response window is left at fill_value.
    for i, s in enumerate(samples):
        torch.testing.assert_close(sequence[i][: s["prompt_len"] - 1], torch.zeros(s["prompt_len"] - 1, topk).double())


def test_full_kl_teacher_logprob_remap_round_trips():
    """SDPO full-vocab KL: the co-located teacher's per-response-token log-softmax *vector*
    (shape (bsz, resp_len, vocab/tp)) is remapped onto the student's [prompt | response] layout via
    ``response_to_sequence_nested(..., fill_value=0.0)``, then ``response_from_nested`` must recover
    it exactly. Locks the vector-payload round-trip the co-located teacher forward relies on
    (``_sdpo_teacher_full_log_probs``), where the trailing dim is the full vocab shard, not top-k.
    """
    samples = [{"prompt_len": 4, "resp_len": 3}, {"prompt_len": 2, "resp_len": 5}]
    vocab_shard = 7  # stands in for V/tp
    prompt_lens = torch.tensor([s["prompt_len"] for s in samples])
    response_lens = torch.tensor([s["resp_len"] for s in samples])
    max_resp = int(response_lens.max())

    padded = torch.full((len(samples), max_resp, vocab_shard), -123.0, dtype=torch.float64)
    for i, s in enumerate(samples):
        for j in range(s["resp_len"]):
            padded[i, j] = torch.arange(vocab_shard, dtype=torch.float64) + 1000 * i + 10 * j

    sequence = response_to_sequence_nested(padded, prompt_lens, response_lens, fill_value=0.0)
    torch.testing.assert_close(sequence.offsets().diff(), (prompt_lens + response_lens).to(torch.int64))

    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(s["resp_len"], dtype=torch.float64) for s in samples], layout=torch.jagged
    )
    recovered = response_from_nested(sequence, response_mask)
    for i, s in enumerate(samples):
        torch.testing.assert_close(recovered[i], padded[i, : s["resp_len"]])
    # Prompt region filled with 0.0 (masked out downstream by response_mask in the loss aggregation).
    for i, s in enumerate(samples):
        torch.testing.assert_close(
            sequence[i][: s["prompt_len"] - 1], torch.zeros(s["prompt_len"] - 1, vocab_shard).double()
        )
