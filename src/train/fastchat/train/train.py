# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adopted from lm-sys@FastChat. Changes include:
# 1. adapt to our NER conversation template.
# 2. change the model saving code to avoid OOM.
# Below is the original copyright:
#    Copyright 2023 Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import random

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, LlamaTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


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


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == -1:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    # """Collects the state dict and dump to disk."""
    # state_dict = trainer.model.state_dict()
    # if trainer.args.should_save:
    #     cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    #     del state_dict
    #     trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
    model = trainer.model
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
        if trainer.args.should_save:
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def tokenize(tokenizer, prompt, add_special_tokens=False):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )
    return result.input_ids


def preprocess2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("ie_as_qa")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    input_ids, labels = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        

        conv_input_ids, conv_targets = [], []

        # Randomly shuffle the queries, since the answer should be order-insensitive to queries.
        assert len(source) % 2 == 0
        idxs = list(range(len(source) // 2))[1:]
        random.shuffle(idxs)
        idxs = [0] + idxs
        new_source = []
        for rand_i in idxs:
            new_source.append(source[2 * rand_i])
            new_source.append(source[2 * rand_i + 1])
        source = new_source

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if j == 0:
                # first input consists of system message and user input
                message = conv.system + conv.sep + role + ": " + sentence["value"]
                _input_ids = tokenize(tokenizer, message, True)
                conv_input_ids += _input_ids
                conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
            else:
                if j % 2 == 0:
                    # user input
                    message = role + ": " + sentence["value"]
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
                else:
                    # assistant output
                    message = role + ": "
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
                    message = sentence["value"] + conv.sep2
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += _input_ids

        assert len(conv_input_ids) == len(conv_targets), source

        if len(conv_input_ids) < tokenizer.model_max_length:
            pad_length = tokenizer.model_max_length - len(conv_input_ids)
            conv_input_ids += [tokenizer.pad_token_id] * pad_length
            conv_targets += [IGNORE_TOKEN_ID] * pad_length

        input_ids.append(conv_input_ids[:tokenizer.model_max_length])
        labels.append(conv_targets[:tokenizer.model_max_length])

        if False:
            cur_user_ids, cur_gpt_ids = [], []
            for i, t in zip(conv_input_ids, conv_targets):
                if t == IGNORE_TOKEN_ID:
                    if i != tokenizer.pad_token_id:
                        cur_user_ids.append(i)
                    if len(cur_gpt_ids):
                        rank0_print(f"OUTPUT: '{tokenizer.decode(cur_gpt_ids)}'")
                        cur_gpt_ids = []
                else:
                    assert i == t, breakpoint()
                    cur_gpt_ids.append(t)
                    if len(cur_user_ids):
                        rank0_print(f"INPUT: '{tokenizer.decode(cur_user_ids)}'")
                        cur_user_ids = []
            assert len(cur_user_ids) == 0, breakpoint()
            if len(cur_gpt_ids):
                rank0_print(f"OUTPUT: '{tokenizer.decode(cur_gpt_ids)}'")
            breakpoint()

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess2(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess2([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    np.random.seed(3)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = 0

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
