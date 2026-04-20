from copy import deepcopy

import examples.gpt_2 as gpt2_module
from autograd import functional, nn
from autograd.backend import xp
from autograd.tensor import Tensor, checkpoint, no_grad
from examples.gpt_2 import GPT2
from test.helpers import array_equal


def _build_tokens(*, batch_size: int, seq_len: int, vocab_size: int):
    input_ids = xp.random.randint(0, vocab_size, (batch_size, seq_len), dtype=xp.int32)
    labels = xp.random.randint(0, vocab_size, (batch_size, seq_len), dtype=xp.int32)
    return input_ids, labels


def _record_decoder_outputs(model: GPT2):
    decoder_outputs = {}
    original_forwards = []

    for idx, sublayer in enumerate(model.sublayers):
        original_forward = sublayer.forward
        original_forwards.append((sublayer, original_forward))

        def _wrapped_forward(x, *, _idx=idx, _forward=original_forward):
            out = _forward(x)
            decoder_outputs.setdefault(_idx, xp.array(out.data))
            return out

        sublayer.forward = _wrapped_forward

    return decoder_outputs, original_forwards


def _restore_decoder_forwards(original_forwards):
    for sublayer, original_forward in original_forwards:
        sublayer.forward = original_forward


def _run_training_step(
    *,
    state_dict,
    input_ids,
    labels,
    dropout_prob: float,
    activation_checkpointing: bool,
):
    model = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=input_ids.shape[1],
        dropout_prob=dropout_prob,
        num_decoder_layers=3,
        activation_checkpointing=activation_checkpointing,
    )
    model.load_state_dict(deepcopy(state_dict))
    model.train()
    decoder_outputs, original_forwards = _record_decoder_outputs(model)
    try:
        logits = model(Tensor(input_ids, requires_grad=False))
        loss = functional.cross_entropy(
            logits,
            labels,
            label_smoothing=0.0,
        )
        loss.backward()
    finally:
        _restore_decoder_forwards(original_forwards)

    grads = {
        name: xp.array(param.grad.data)
        for name, param in model.parameters.items()
        if param.grad is not None
    }
    return {
        "decoder_outputs": decoder_outputs,
        "logits": xp.array(logits.data),
        "loss": xp.array(loss.data),
        "grads": grads,
    }


def _compare_training_runs(reference, candidate):
    assert reference["decoder_outputs"].keys() == candidate["decoder_outputs"].keys()
    for idx in reference["decoder_outputs"]:
        assert array_equal(
            reference["decoder_outputs"][idx], candidate["decoder_outputs"][idx]
        ), f"decoder block {idx} output mismatch"

    assert array_equal(reference["logits"], candidate["logits"])
    assert array_equal(reference["loss"], candidate["loss"])
    assert reference["grads"].keys() == candidate["grads"].keys()
    for name in reference["grads"]:
        assert array_equal(reference["grads"][name], candidate["grads"][name]), (
            f"gradient mismatch for {name}"
        )


def test_gpt2_activation_checkpointing_matches_baseline_without_dropout():
    xp.random.seed(7)
    input_ids, labels = _build_tokens(batch_size=2, seq_len=8, vocab_size=32)
    xp.random.seed(11)
    baseline_model = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=3,
        activation_checkpointing=False,
    )
    state_dict = deepcopy(baseline_model.state_dict())

    xp.random.seed(17)
    baseline = _run_training_step(
        state_dict=state_dict,
        input_ids=input_ids,
        labels=labels,
        dropout_prob=0.0,
        activation_checkpointing=False,
    )
    xp.random.seed(17)
    checkpointed = _run_training_step(
        state_dict=state_dict,
        input_ids=input_ids,
        labels=labels,
        dropout_prob=0.0,
        activation_checkpointing=True,
    )

    _compare_training_runs(baseline, checkpointed)


def test_gpt2_activation_checkpointing_matches_baseline_with_dropout():
    xp.random.seed(23)
    input_ids, labels = _build_tokens(batch_size=2, seq_len=8, vocab_size=32)
    xp.random.seed(29)
    baseline_model = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=8,
        dropout_prob=0.2,
        num_decoder_layers=3,
        activation_checkpointing=False,
    )
    state_dict = deepcopy(baseline_model.state_dict())

    xp.random.seed(31)
    baseline = _run_training_step(
        state_dict=state_dict,
        input_ids=input_ids,
        labels=labels,
        dropout_prob=0.2,
        activation_checkpointing=False,
    )
    xp.random.seed(31)
    checkpointed = _run_training_step(
        state_dict=state_dict,
        input_ids=input_ids,
        labels=labels,
        dropout_prob=0.2,
        activation_checkpointing=True,
    )

    _compare_training_runs(baseline, checkpointed)


def test_gpt2_activation_checkpointing_matches_baseline_in_eval_mode():
    xp.random.seed(37)
    input_ids, _ = _build_tokens(batch_size=2, seq_len=8, vocab_size=32)
    xp.random.seed(41)
    baseline_model = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=8,
        dropout_prob=0.2,
        num_decoder_layers=3,
        activation_checkpointing=False,
    )
    state_dict = deepcopy(baseline_model.state_dict())

    baseline = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=8,
        dropout_prob=0.2,
        num_decoder_layers=3,
        activation_checkpointing=False,
    )
    baseline.load_state_dict(deepcopy(state_dict))
    baseline.eval()

    checkpointed = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=8,
        dropout_prob=0.2,
        num_decoder_layers=3,
        activation_checkpointing=True,
    )
    checkpointed.load_state_dict(deepcopy(state_dict))
    checkpointed.eval()

    baseline_logits = baseline(Tensor(input_ids, requires_grad=False))
    checkpointed_logits = checkpointed(Tensor(input_ids, requires_grad=False))

    assert array_equal(baseline_logits.data, checkpointed_logits.data)


def test_checkpoint_helper_matches_baseline_with_dropout():
    x_data = xp.arange(12, dtype=xp.float32).reshape(3, 4) / 10.0
    weight_data = xp.arange(16, dtype=xp.float32).reshape(4, 4) / 20.0
    bias_data = xp.arange(4, dtype=xp.float32) / 30.0

    def build_inputs():
        x = Tensor(x_data, requires_grad=True)
        weight = Tensor(weight_data, requires_grad=True)
        bias = Tensor(bias_data, requires_grad=True)
        return x, weight, bias

    dropout = nn.Dropout(p=0.25)
    dropout.train()

    def block(x, weight, bias):
        out = (x @ weight) + bias
        return dropout(out)

    xp.random.seed(43)
    x_ref, weight_ref, bias_ref = build_inputs()
    ref_out = block(x_ref, weight_ref, bias_ref)
    ref_loss = ref_out.sum()
    ref_loss.backward()

    xp.random.seed(43)
    x_ckpt, weight_ckpt, bias_ckpt = build_inputs()
    ckpt_out = checkpoint(block, x_ckpt, weight_ckpt, bias_ckpt)
    ckpt_loss = ckpt_out.sum()
    ckpt_loss.backward()

    assert array_equal(ref_out.data, ckpt_out.data)
    assert array_equal(x_ref.grad.data, x_ckpt.grad.data)
    assert array_equal(weight_ref.grad.data, weight_ckpt.grad.data)
    assert array_equal(bias_ref.grad.data, bias_ckpt.grad.data)


def test_checkpoint_passthrough_when_grad_disabled():
    x = Tensor(xp.arange(4, dtype=xp.float32), requires_grad=True)

    with no_grad():
        out = checkpoint(lambda t: t + 1, x)

    assert array_equal(out.data, x.data + 1)
    assert out.creator is None
    assert not out.requires_grad


def test_gpt2_training_uses_checkpoint_helper_when_enabled(monkeypatch):
    call_count = 0

    def counting_checkpoint(run_function, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return run_function(*args)

    monkeypatch.setattr(gpt2_module, "checkpoint", counting_checkpoint)

    xp.random.seed(47)
    model = GPT2(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=3,
        activation_checkpointing=True,
    )
    model.train()

    logits = model(Tensor(xp.random.randint(0, 32, (2, 8), dtype=xp.int32)))

    assert logits.shape == (2, 8, 32)
    assert call_count == 3
