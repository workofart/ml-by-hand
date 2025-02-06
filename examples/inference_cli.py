import argparse
import logging
import sys

try:
    import cupy as np  # type: ignore
    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

from autograd.text.tokenizer import BytePairEncoder
from autograd.tools.trainer import LLMTrainer
from autograd.text import utils as text_utils
from autograd import optim, functional, nn
from autograd.tools.config_schema import TransformerTrainingConfig, CustomBpeConfig
import importlib
gpt2 = importlib.import_module("examples.gpt-2")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple inference script for a GPT-style model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file to load the model from"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["chat", "sampling"],
        help="Inference mode: 'chat' or 'sampling'."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate in 'sampling' mode."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k cutoff for sampling."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max tokens to generate."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="The prompt to start the conversation in 'sampling' mode"
    )
    return parser.parse_args()


TRAINING_DATA_MAP = {
    "wiki_8000": {
        "vocab_size": 8260,
        "num_merges": 8000,
        "encoded_data_path": "training_data/bpe_8000_wiki_simple_encoded_data",
        "vocab_path": "training_data/wikipedia_simpleenglish_vocab_8000.pkl",
    },
    "wiki_12000": {
        "vocab_size": 12260,
        "num_merges": 12000,
        "encoded_data_path": "training_data/bpe_12000_wiki_simple_encoded_data",
        "vocab_path": "training_data/wikipedia_simpleenglish_vocab_12000.pkl",
    },
    "shakespeare_0": {
        "vocab_size": 260,
        "num_merges": 0,
        "encoded_data_path": "training_data/bpe_0_shakespeare_encoded_data.npz",
        "vocab_path": "training_data/shakespeare_vocab_0.pkl",
    },
    "shakespeare_3000": {
        "vocab_size": 4260,
        "num_merges": 3000,
        "encoded_data_path": "training_data/bpe_3000_shakespeare_encoded_data.npz",
        "vocab_path": "training_data/shakespeare_vocab_3000.pkl",
    },
    "openwebtext_tiktoken": {
        "vocab_size": 50257,
        # "num_merges": 0,
        # "encoded_data_path": "training_data/openwebtext_tiktoken_encoded_data",
        # "vocab_path": "training_data/openwebtext_tiktoken_vocab.pkl"
    },
}

def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    
    dataset_name = "wiki_8000"

    # --------------------------------------------------------------------
    # 1. Load the checkpoint (including model, optimizer, bpe, config etc.)
    # --------------------------------------------------------------------
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    
    # dummy config
    config = TransformerTrainingConfig(
        total_epochs=10,
        checkpoint_freq=0,
        model_kwargs={
            "vocab_size": TRAINING_DATA_MAP[dataset_name]["vocab_size"],
            # TODO: completely remove reliance on these dummy config values, check checkpoint loading logic
            # "max_seq_len": 740,  # wiki
            # "max_seq_len": 392,  # openwebtext
            "max_seq_len": 480,  # shakespeare
        },
        optimizer_kwargs={
            "lr": 1e-3,
        },
        teacher_enforcing=False,
        include_decoder_input=False,
        create_padding_masks=False,
        label_smoothing=0.0,
        custom_bpe=CustomBpeConfig(
            num_merges=TRAINING_DATA_MAP[dataset_name]["num_merges"],
            encoded_data_path=TRAINING_DATA_MAP[dataset_name]["encoded_data_path"],
            vocab_path=TRAINING_DATA_MAP[dataset_name]["vocab_path"],
            overwrite_encoded_data=False,
            overwrite_vocabulary_file=False,
            split_token="<|endoftext|>",
        )
    )
    bpe = BytePairEncoder(
       num_merges=config.custom_bpe.num_merges,
       vocab_file_path=config.custom_bpe.vocab_path,
       encoded_data_path=config.custom_bpe.encoded_data_path,
    )
    trainer = LLMTrainer(
        model_cls=gpt2.GPT2,
        optimizer_cls=optim.Adam,
        loss_fn=functional.cross_entropy,
        config=config,
        forward_fn=gpt2.GPT2ForwardFn(),
        checkpoint_path=args.checkpoint,
    )

    logger.info("Model and BPE loaded successfully.")

    # ---------------------------------------------------------
    # 2. Choose mode and run inference
    # ---------------------------------------------------------

    if args.mode == "chat":
        logger.info("Entering chat mode. Type 'quit' or 'exit' to stop.")
        while True:
            user_input = input("\n >> User: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                logger.info("Exiting chat mode.")
                break

            text_utils.inference(
                model=trainer.model,
                prediction_func=gpt2.GPT2ForwardFn(),
                bpe=bpe,
                start_tokens=user_input,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
            )

    elif args.mode == "sampling":
        logger.info(f"Generating {args.num_samples} sample(s).")
        for i in range(args.num_samples):
            text_utils.inference(
                model=trainer.model,
                prediction_func=gpt2.GPT2ForwardFn(),
                bpe=bpe,
                start_tokens=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        logger.info("Sampling complete. Exiting.")


if __name__ == "__main__":
    main()
