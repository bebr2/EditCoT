import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

from typing import Optional
from dataclasses import dataclass, field
import transformers
from transformers.training_args import TrainingArguments
from dataset import SupervisedDataset
import torch
from transformers.trainer_utils import RemoveColumnsCollator

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="none")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()