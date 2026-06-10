import os

from autograd import functional, optim
from autograd.data.collator import BatchMaxLengthCausalLMCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import MapDataset
from autograd.data.sampler import TokenLengthGroupedRandomSampler
from autograd.data.sft import (
    SFT_ROLE_MARKERS,
    SFT_SYSTEM_PROMPT,
    SFT_TURN_SEPARATOR,
    load_gsm8k_grpo_sft,
    load_no_robots_sft,
    prepare_sft_token_sequences,
)
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate_text
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


def format_sft_prompt(user_content: str, *, system_content: str | None = None) -> str:
    prefix = ""
    if system_content is not None:
        prefix = f"{SFT_ROLE_MARKERS['system']}{system_content}{SFT_TURN_SEPARATOR}"
    return (
        f"{prefix}{SFT_ROLE_MARKERS['user']}{user_content}"
        f"{SFT_TURN_SEPARATOR}{SFT_ROLE_MARKERS['assistant']}"
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    from gpt_2 import GPT2, GPT2ForwardFn

    # Supported: "gsm8k", "no_robots".
    SFT_DATASET = "gsm8k"

    if SFT_DATASET == "gsm8k":
        load_sft_split = load_gsm8k_grpo_sft
        training_run_name = f"sft_{SFT_DATASET}_grpo_system"
        encoded_dataset_name = f"{SFT_DATASET}_grpo_system"
        eval_start_string = format_sft_prompt(
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?",
            system_content=SFT_SYSTEM_PROMPT,
        )
    elif SFT_DATASET == "no_robots":
        load_sft_split = load_no_robots_sft
        training_run_name = f"sft_{SFT_DATASET}"
        encoded_dataset_name = SFT_DATASET
        eval_start_string = format_sft_prompt("What is the weather today?")
    else:
        raise ValueError("SFT_DATASET must be one of: gsm8k, no_robots")

    CONFIG = TransformerTrainingConfig(
        training_run_name=training_run_name,
        dataset_name=SFT_DATASET,
        max_steps=900,
        max_eval_steps=20,
        checkpoint_freq=300,
        report_every_steps=50,
        global_batch_size=32,
        micro_batch_size=4,
        eval_batch_size=4,
        model_kwargs={
            # Architecture must match the openwebtext_gpt2_124m_baseline
            # pretraining checkpoint loaded via pretrained_checkpoint_path.
            "num_attention_heads": 12,
            "hidden_size": 768,
            # Padded vocab (next multiple of 64 above GPT-2's 50,257) for
            # efficient matmul; matches the pretraining config.
            "vocab_size": 50_304,
            # dropout=0 matches the GPT-2 paper's finetuning recipe.
            "dropout_prob": 0.0,
            "max_seq_len": 1024,
            "num_decoder_layers": 12,
            "activation_checkpointing": False,
            "parameter_dtype": "bfloat16",
        },
        optimizer_kwargs={
            "lr": 5e-5,
            "beta2": 0.99,
            "weight_decay": 0.1,
            "lr_scheduler_kwargs": {
                "lr_scheduler_cls": optim.CosineScheduler,
                "warmup_steps": 90,  # 15% of max_steps
                "lr_decay_iters": 720,  # 80% of max_steps
            },
        },
        max_grad_norm=1.0,
        # Basename without .json/.npz. The configured model architecture below
        # must match this checkpoint; load_state_dict will fail otherwise.
        pretrained_checkpoint_path=project_path(
            "checkpoints", "openwebtext_gpt2_124m_baseline_GPT2_14000"
        ),
        label_smoothing=0.1,
        teacher_forcing=False,
        eval_start_string=eval_start_string,
        custom_bpe=CustomBpeConfig(
            num_merges=49990,
            encoded_data_path=project_path(
                "training_data",
                f"bpe_49990_{encoded_dataset_name}_encoded_data_sft.npz",
            ),
            vocab_path=project_path("training_data", "openwebtext_vocab_49990.pkl"),
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            start_token="<SOS>",
            split_token="<|endoftext|>",
        ),
    )

    train_chat_examples = load_sft_split(split="train")
    val_chat_examples = load_sft_split(split="test")
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

    # Model embeds the padded vocab; BPE token IDs must fit within it.
    if bpe.n_vocab > CONFIG.model_kwargs["vocab_size"]:
        raise ValueError(
            f"BPE vocab ({bpe.n_vocab}) exceeds model vocab_size "
            f"({CONFIG.model_kwargs['vocab_size']})."
        )

    def generate_eval_samples(
        model,
        _forward_fn,
        _val_data_loader,
        config: TransformerTrainingConfig,
    ) -> None:
        generate_text(
            model=model,
            prediction_func=GPT2ForwardFn(),
            bpe=bpe,
            start_tokens=config.eval_start_string,
            max_length=min(256, int(model.max_seq_len)),
            temperature=0.8,
            top_k=config.eval_top_k,
            stop_token=SFT_TURN_SEPARATOR,
        )

    trainer = LLMTrainer(
        model_cls=GPT2,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=CONFIG,
        forward_fn=GPT2ForwardFn(),
        eval_callbacks=[generate_eval_samples],
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
    train_data_loader = build_data_loader(
        train_examples,
        batch_size=CONFIG.micro_batch_size,
        max_tokens=max_tokens,
        pad_idx=pad_idx,
        sort_buffer_size=CONFIG.global_batch_size,
    )
    val_data_loader = build_data_loader(
        val_examples,
        batch_size=CONFIG.eval_batch_size,
        max_tokens=max_tokens,
        pad_idx=pad_idx,
        sort_buffer_size=CONFIG.eval_batch_size,
    )

    trainer.fit(train_data_loader, val_data_loader)

    # Inference test
    for k in range(5):
        generate_text(
            model=trainer.model,
            prediction_func=GPT2ForwardFn(),
            bpe=bpe,
            start_tokens=CONFIG.eval_start_string,
            max_length=int(trainer.model.max_seq_len),
            temperature=0.8,
            top_k=CONFIG.eval_top_k,
            stop_token=SFT_TURN_SEPARATOR,
        )
        print("\n------------------------\n")
