"""
Per-user temporal holdout split.

Strategy:
  - last item       → test  target
  - second-last     → val   target
  - everything else → training sequence
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from src.data.dataset import SequenceDataset, Example


def split_user_sequences(
    user_sequences: Dict[int, List[Tuple[int, int]]]
) -> Tuple[
    Dict[int, List[Tuple[int, int]]],
    Dict[int, Tuple[List[int], int, int]],
    Dict[int, Tuple[List[int], int, int]],
]:
    train_seqs: Dict[int, List[Tuple[int, int]]] = {}
    val_data: Dict[int, Tuple[List[int], int, int]] = {}
    test_data: Dict[int, Tuple[List[int], int, int]] = {}

    for user_idx, seq in user_sequences.items():
        if len(seq) < 3:
            continue

        train_part = seq[:-2]
        val_item, val_tb = seq[-2]
        test_item, test_tb = seq[-1]

        train_seqs[user_idx] = train_part

        val_input = [item for item, _ in train_part]
        val_data[user_idx] = (val_input, val_item, val_tb)

        test_input = [item for item, _ in seq[:-1]]
        test_data[user_idx] = (test_input, test_item, test_tb)

    return train_seqs, val_data, test_data


def build_train_examples(
    train_seqs: Dict[int, List[Tuple[int, int]]]
) -> List[Example]:
    examples: List[Example] = []
    for user_idx, seq in train_seqs.items():
        if len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            input_items = [item for item, _ in seq[:t]]
            target_item, target_tb = seq[t]
            examples.append((user_idx, input_items, target_item, target_tb))
    return examples


def build_eval_examples(
    eval_data: Dict[int, Tuple[List[int], int, int]]
) -> List[Example]:
    examples: List[Example] = []
    for user_idx, (input_items, target_item, target_tb) in eval_data.items():
        examples.append((user_idx, input_items, target_item, target_tb))
    return examples


def make_datasets(
    user_sequences: Dict[int, List[Tuple[int, int]]],
    max_seq_len: int = 50,
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset,
           Dict[int, List[Tuple[int, int]]]]:
    train_seqs, val_data, test_data = split_user_sequences(user_sequences)

    train_ds = SequenceDataset(build_train_examples(train_seqs), max_seq_len)
    val_ds = SequenceDataset(build_eval_examples(val_data), max_seq_len)
    test_ds = SequenceDataset(build_eval_examples(test_data), max_seq_len)

    return train_ds, val_ds, test_ds, train_seqs
