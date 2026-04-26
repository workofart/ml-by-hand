import os

from autograd import functional, optim
from autograd.data.collator import BatchMaxLengthCausalLMCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import MapDataset
from autograd.data.sampler import TokenLengthGroupedRandomSampler
from autograd.data.sft import load_no_robots_sft, prepare_sft_token_sequences
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.callback import run_sampling_inference
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from autograd.tools.trainer import LLMTrainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def filter_fitting_examples(token_sequences, loss_masks, max_tokens: int):
    return [
        {"tokens": tokens, "loss_mask": loss_mask}
        for tokens, loss_mask in zip(token_sequences, loss_masks)
        if len(tokens) <= max_tokens
    ]


def build_data_loader(
    examples,
    *,
    batch_size: int,
    max_tokens: int,
    pad_idx: int,
    sort_buffer_size: int,
) -> DataLoader:
    dataset = MapDataset(examples)
    collator = BatchMaxLengthCausalLMCollator(max_tokens=max_tokens, pad_idx=pad_idx)
    sampler = TokenLengthGroupedRandomSampler(
        dataset, sort_buffer_size=sort_buffer_size
    )
    return DataLoader(dataset, batch_size, collator, sampler=sampler)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    from gpt_2 import GPT2, GPT2ForwardFn

    CONFIG = TransformerTrainingConfig(
        training_run_name="sft_demo",
        dataset_name="no_robots",
        max_steps=500,
        max_eval_steps=10,
        checkpoint_freq=32,
        global_batch_size=32,
        micro_batch_size=4,
        model_kwargs={
            "num_attention_heads": 12,  # GPT-2 small uses 12
            "hidden_size": 1536,  # GPT-2 small uses 768, must be divisible by num_attention_heads
            "dropout_prob": 0.2,
            "max_seq_len": 1024,  # GPT-2 uses 1024
            "num_decoder_layers": 12,  # GPT-2 uses 12
            "activation_checkpointing": True,
        },
        optimizer_kwargs={
            "lr": 1e-4,
            "beta2": 0.99,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 75,  # 15% of max_steps
                "lr_decay_iters": 400,  # 80% of max_steps
            },
        },
        max_grad_norm=1.0,
        # Basename without .json/.npz. The configured model architecture below
        # must match this checkpoint; load_state_dict will fail otherwise.
        pretrained_checkpoint_path=project_path("checkpoints", "wiki_GPT2_62000"),
        label_smoothing=0.1,
        teacher_forcing=False,
        eval_start_string="User: What is the weather today?<|endoftext|>Assistant: ",
        custom_bpe=CustomBpeConfig(
            num_merges=12000,
            encoded_data_path=project_path(
                "training_data", "bpe_12000_no_robots_encoded_data_sft.npz"
            ),
            vocab_path=project_path(
                "training_data", "wikipedia_simpleenglish_vocab_12000.pkl"
            ),
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            split_token="<|endoftext|>",
        ),
    )

    train_chat_examples = load_no_robots_sft(split="train")
    val_chat_examples = load_no_robots_sft(split="test")
    bpe_config = CONFIG.custom_bpe
    if bpe_config is None:
        raise ValueError(
            "Please supply a custom_bpe config. Check out CustomBpeConfig for more details."
        )
    if not os.path.exists(bpe_config.vocab_path):
        raise FileNotFoundError(f"Expected pretrained vocab at {bpe_config.vocab_path}")

    bpe = BytePairEncoder(
        num_merges=bpe_config.num_merges,
        vocab_file_path=bpe_config.vocab_path,
        encoded_data_path=bpe_config.encoded_data_path,
    )

    CONFIG.model_kwargs["vocab_size"] = bpe.n_vocab

    trainer = LLMTrainer(
        model_cls=GPT2,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT2ForwardFn(),
    )

    pad_idx = bpe.encode("<PAD>")[0]
    max_tokens = trainer.model.max_seq_len + 1
    split = len(train_chat_examples)
    token_sequences, loss_masks = prepare_sft_token_sequences(
        train_chat_examples + val_chat_examples,
        bpe,
        overwrite_encoded_data=bpe_config.overwrite_encoded_data,
        desc="Tokenizing SFT examples",
    )

    train_examples = filter_fitting_examples(
        token_sequences[:split],
        loss_masks[:split],
        max_tokens,
    )
    val_examples = filter_fitting_examples(
        token_sequences[split:],
        loss_masks[split:],
        max_tokens,
    )
    if not train_examples or not val_examples:
        raise ValueError("No SFT examples fit within the configured context window.")

    print(
        "Data length: "
        f"raw_train={split} raw_val={len(token_sequences) - split} "
        f"fit_train={len(train_examples)} fit_val={len(val_examples)}"
    )
    val_batch_size = max(1, CONFIG.micro_batch_size // 2)
    train_data_loader = build_data_loader(
        train_examples,
        batch_size=CONFIG.micro_batch_size,
        max_tokens=max_tokens,
        pad_idx=pad_idx,
        sort_buffer_size=CONFIG.global_batch_size,
    )
    val_data_loader = build_data_loader(
        val_examples,
        batch_size=val_batch_size,
        max_tokens=max_tokens,
        pad_idx=pad_idx,
        sort_buffer_size=val_batch_size,
    )

    trainer.fit(train_data_loader, val_data_loader)

    # Inference test
    for k in range(5):
        run_sampling_inference(
            model=trainer.model,
            forward_fn=GPT2ForwardFn(),
            bpe=bpe,
            start_tokens=CONFIG.eval_start_string,
            max_length=int(trainer.model.max_seq_len),
            top_k=CONFIG.eval_top_k,
        )
        print("\n------------------------\n")
