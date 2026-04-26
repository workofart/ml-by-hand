import os

from autograd import functional, optim
from autograd.backend import xp
from autograd.data.collator import BatchMaxLengthCausalLMCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import PairedMapDataset
from autograd.data.sampler import TokenLengthGroupedRandomSampler
from autograd.data.sft import load_no_robots_sft, prepare_sft_token_sequences
from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.callback import run_sampling_inference
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from autograd.tools.trainer import LLMTrainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


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
        eval_start_string="What is the weather today?",
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
    chat_examples = train_chat_examples + val_chat_examples

    if CONFIG.custom_bpe:
        # Reuse the pretrained tokenizer vocabulary directly for SFT.
        if not os.path.exists(CONFIG.custom_bpe.vocab_path):
            raise FileNotFoundError(
                f"Expected pretrained vocab at {CONFIG.custom_bpe.vocab_path}"
            )
        bpe = BytePairEncoder(
            num_merges=CONFIG.custom_bpe.num_merges,
            vocab_file_path=CONFIG.custom_bpe.vocab_path,
            encoded_data_path=CONFIG.custom_bpe.encoded_data_path,
        )
    else:
        raise ValueError(
            "Please supply a custom_bpe config. Check out CustomBpeConfig for more details."
        )

    CONFIG.model_kwargs["vocab_size"] = bpe.n_vocab

    trainer = LLMTrainer(
        model_cls=GPT2,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT2ForwardFn(),
    )

    pad_idx = bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
    max_tokens = trainer.model.max_seq_len + 1
    token_sequences, loss_masks = prepare_sft_token_sequences(
        chat_examples,
        bpe,
        overwrite_encoded_data=CONFIG.custom_bpe.overwrite_encoded_data,
        desc="Tokenizing SFT examples",
    )
    train_token_sequences = token_sequences[: len(train_chat_examples)]
    train_loss_masks = loss_masks[: len(train_chat_examples)]
    val_token_sequences = token_sequences[len(train_chat_examples) :]
    val_loss_masks = loss_masks[len(train_chat_examples) :]

    train_examples = [
        {"tokens": tokens, "loss_mask": loss_mask}
        for tokens, loss_mask in zip(train_token_sequences, train_loss_masks)
        if len(tokens) <= max_tokens
    ]
    val_examples = [
        {"tokens": tokens, "loss_mask": loss_mask}
        for tokens, loss_mask in zip(val_token_sequences, val_loss_masks)
        if len(tokens) <= max_tokens
    ]
    if not train_examples or not val_examples:
        raise ValueError("No SFT examples fit within the configured context window.")

    fit_train = len(train_examples)
    fit_val = len(val_examples)
    print(
        "Data length: "
        f"raw_train={len(train_token_sequences)} raw_val={len(val_token_sequences)} "
        f"fit_train={fit_train} fit_val={fit_val}"
    )
    val_batch_size = max(1, CONFIG.micro_batch_size // 2)
    train_dataset = PairedMapDataset(
        [example["tokens"] for example in train_examples],
        [example["loss_mask"] for example in train_examples],
        input_key="tokens",
        target_key="loss_mask",
        dtype=xp.int32,
    )
    val_dataset = PairedMapDataset(
        [example["tokens"] for example in val_examples],
        [example["loss_mask"] for example in val_examples],
        input_key="tokens",
        target_key="loss_mask",
        dtype=xp.int32,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.micro_batch_size,
        collator=BatchMaxLengthCausalLMCollator(
            max_tokens=max_tokens,
            pad_idx=pad_idx,
        ),
        sampler=TokenLengthGroupedRandomSampler(
            train_dataset,
            sort_buffer_size=CONFIG.global_batch_size,
        ),
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        collator=BatchMaxLengthCausalLMCollator(
            max_tokens=max_tokens,
            pad_idx=pad_idx,
        ),
        sampler=TokenLengthGroupedRandomSampler(
            val_dataset,
            sort_buffer_size=val_batch_size,
        ),
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
