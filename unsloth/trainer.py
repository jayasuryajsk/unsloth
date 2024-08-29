# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from trl import SFTTrainer
from . import is_bfloat16_supported

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
]

@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Different learning rates for embeddings and lm_head."}
    )
    use_streaming_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming dataset for training."}
    )

def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr=5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)
    param_groups = {
        "non_embeddings": {},
        "embeddings": {},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["non_embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer

    def get_train_dataloader(self):
        if self.args.use_streaming_dataset:
            return self._get_streaming_train_dataloader()
        return super().get_train_dataloader()

    def _get_streaming_train_dataloader(self):
        from datasets import IterableDataset
        from torch.utils.data import DataLoader

        if not isinstance(self.train_dataset, IterableDataset):
            raise ValueError("Training dataset must be an instance of IterableDataset when use_streaming_dataset is True")

        data_collator = self.data_collator
        train_dataset = self.train_dataset

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        return DataLoader(train_dataset, **dataloader_params)

    def train(self, resume_from_checkpoint=None, **kwargs):
        if self.args.use_streaming_dataset:
            # Override max_steps if it's not set and we're using a streaming dataset
            if self.args.max_steps == -1:
                raise ValueError("When using a streaming dataset, max_steps must be set to a positive integer.")

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
