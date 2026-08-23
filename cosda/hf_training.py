from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import json
import os
import random
import shutil

from .data import CLASSIFICATION_TASKS, dataset_entry, iter_records, load_manifest
from .evaluate import load_selected_as_records
from .io_utils import iter_jsonl, write_json
from .types import DataRecord


def train_hf(
    manifest_path: str,
    dataset_id: str,
    language: str,
    gold_path: str | Path,
    selected_path: str | Path | None,
    output_dir: str | Path,
    model_name: str,
    epochs: float = 3.0,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    keep_checkpoints: bool = False,
    seed: int = 13,
    deterministic: bool | None = None,
) -> dict:
    deterministic = env_flag("COSDA_HF_DETERMINISTIC") if deterministic is None else deterministic
    manifest = load_manifest(manifest_path)
    task = dataset_entry(manifest, dataset_id=dataset_id)["task"]
    gold = [DataRecord.from_json(row) for row in iter_jsonl(gold_path)]
    synthetic = load_selected_as_records(selected_path) if selected_path else []
    train = stable_training_order(gold + synthetic, seed)
    dev = list(iter_records(manifest, dataset_id, language, "dev"))
    test = list(iter_records(manifest, dataset_id, language, "test"))

    if task in CLASSIFICATION_TASKS:
        return train_hf_classification(
            train, dev, test, output_dir, model_name, epochs, batch_size, lr, max_length, keep_checkpoints, seed, deterministic
        )
    if task == "named_entity_recognition":
        return train_hf_ner(
            train, dev, test, output_dir, model_name, epochs, batch_size, lr, max_length, keep_checkpoints, seed, deterministic
        )
    if task == "summarization":
        return train_hf_summarization(
            train, dev, test, output_dir, model_name, epochs, batch_size, lr, max_length, keep_checkpoints, seed, deterministic
        )
    raise ValueError(f"Unsupported HF training task: {task}")


def stable_training_order(records: list[DataRecord], seed: int) -> list[DataRecord]:
    ordered = sorted(records, key=lambda record: stable_record_key(record))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def stable_record_key(record: DataRecord) -> str:
    payload = "\t".join(
        [
            record.dataset_id,
            record.task,
            record.language,
            record.split,
            record.id,
            str(record.label),
            record.text,
            json.dumps(record.tokens or [], ensure_ascii=False, sort_keys=True),
            json.dumps(record.tags or [], ensure_ascii=False, sort_keys=True),
            record.summary or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends, "cuda"):
                for name, enabled in (
                    ("enable_flash_sdp", False),
                    ("enable_mem_efficient_sdp", False),
                    ("enable_math_sdp", True),
                ):
                    setter = getattr(torch.backends.cuda, name, None)
                    if setter is not None:
                        setter(enabled)
            try:
                torch.use_deterministic_algorithms(
                    True,
                    warn_only=env_flag("COSDA_HF_DETERMINISTIC_WARN_ONLY"),
                )
            except TypeError:
                torch.use_deterministic_algorithms(True)


def precision_kwargs(deterministic: bool = False) -> dict:
    if deterministic:
        return {"bf16": False, "fp16": False, "tf32": False}
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    torch.backends.cuda.matmul.allow_tf32 = True
    use_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {"bf16": use_bf16, "fp16": not use_bf16, "tf32": True}


def train_hf_classification(
    train: list[DataRecord],
    dev: list[DataRecord],
    test: list[DataRecord],
    output_dir: str | Path,
    model_name: str,
    epochs: float,
    batch_size: int,
    lr: float,
    max_length: int,
    keep_checkpoints: bool,
    seed: int,
    deterministic: bool,
) -> dict:
    configure_reproducibility(seed, deterministic)
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    labels = sorted({str(r.label) for r in train + dev + test if r.label is not None})
    label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rows(records: list[DataRecord]):
        return [{"text": r.text, "label": label2id[str(r.label)]} for r in records if r.label is not None]

    train_ds = Dataset.from_list(rows(train))
    dev_ds = Dataset.from_list(rows(dev))
    test_ds = Dataset.from_list(rows(test))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(tok, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label={i: label for label, i in label2id.items()},
    )

    def compute_metrics(pred):
        logits, y = pred
        yhat = logits.argmax(axis=-1)
        return {"accuracy": accuracy_score(y, yhat), "macro_f1": f1_score(y, yhat, average="macro")}

    strategy_kwargs = training_strategy_kwargs(TrainingArguments)
    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        save_strategy="epoch",
        **strategy_kwargs,
        **precision_kwargs(deterministic),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        **training_seed_kwargs(TrainingArguments, seed),
        load_best_model_at_end=True,
        save_total_limit=1,
        dataloader_num_workers=0 if deterministic else 2,
        group_by_length=False if deterministic else True,
        metric_for_best_model="macro_f1",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    write_json(Path(output_dir) / "test_metrics.json", metrics)
    cleanup_checkpoints(output_dir, keep_checkpoints)
    return metrics


def train_hf_ner(
    train: list[DataRecord],
    dev: list[DataRecord],
    test: list[DataRecord],
    output_dir: str | Path,
    model_name: str,
    epochs: float,
    batch_size: int,
    lr: float,
    max_length: int,
    keep_checkpoints: bool,
    seed: int,
    deterministic: bool,
) -> dict:
    configure_reproducibility(seed, deterministic)
    from datasets import Dataset
    from seqeval.metrics import f1_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tag_names = sorted({tag for r in train + dev + test for tag in (r.tags or [])})
    label2id = {tag: i for i, tag in enumerate(tag_names)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rows(records: list[DataRecord]):
        return [{"tokens": r.tokens or r.text.split(), "tags": [label2id[t] for t in (r.tags or [])]} for r in records if r.tags]

    def tokenize_and_align(batch):
        tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, max_length=max_length)
        labels = []
        for i, tags in enumerate(batch["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous = None
            aligned = []
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)
                elif word_id != previous:
                    aligned.append(tags[word_id])
                else:
                    aligned.append(-100)
                previous = word_id
            labels.append(aligned)
        tokenized["labels"] = labels
        return tokenized

    train_ds = Dataset.from_list(rows(train)).map(tokenize_and_align, batched=True, remove_columns=["tokens", "tags"])
    dev_ds = Dataset.from_list(rows(dev)).map(tokenize_and_align, batched=True, remove_columns=["tokens", "tags"])
    test_ds = Dataset.from_list(rows(test)).map(tokenize_and_align, batched=True, remove_columns=["tokens", "tags"])
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag_names),
        label2id=label2id,
        id2label={i: label for label, i in label2id.items()},
    )

    def compute_metrics(pred):
        import numpy as np

        logits, labels = pred
        pred_ids = np.argmax(logits, axis=-1)
        true_tags, pred_tags = [], []
        for label_row, pred_row in zip(labels, pred_ids):
            true_seq, pred_seq = [], []
            for label_id, pred_id in zip(label_row, pred_row):
                if label_id == -100:
                    continue
                true_seq.append(tag_names[label_id])
                pred_seq.append(tag_names[pred_id])
            true_tags.append(true_seq)
            pred_tags.append(pred_seq)
        return {"entity_f1": f1_score(true_tags, pred_tags)}

    strategy_kwargs = training_strategy_kwargs(TrainingArguments)
    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        save_strategy="epoch",
        **strategy_kwargs,
        **precision_kwargs(deterministic),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        **training_seed_kwargs(TrainingArguments, seed),
        load_best_model_at_end=True,
        save_total_limit=1,
        dataloader_num_workers=0 if deterministic else 2,
        metric_for_best_model="entity_f1",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    write_json(Path(output_dir) / "test_metrics.json", metrics)
    cleanup_checkpoints(output_dir, keep_checkpoints)
    return metrics


def train_hf_summarization(
    train: list[DataRecord],
    dev: list[DataRecord],
    test: list[DataRecord],
    output_dir: str | Path,
    model_name: str,
    epochs: float,
    batch_size: int,
    lr: float,
    max_length: int,
    keep_checkpoints: bool,
    seed: int,
    deterministic: bool,
) -> dict:
    configure_reproducibility(seed, deterministic)
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rows(records: list[DataRecord]):
        return [{"text": r.text, "summary": r.summary or ""} for r in records if r.summary]

    def tok(batch):
        inputs = tokenizer(batch["text"], truncation=True, max_length=max_length)
        labels = tokenizer(text_target=batch["summary"], truncation=True, max_length=96)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = Dataset.from_list(rows(train)).map(tok, batched=True, remove_columns=["text", "summary"])
    dev_ds = Dataset.from_list(rows(dev)).map(tok, batched=True, remove_columns=["text", "summary"])
    test_ds = Dataset.from_list(rows(test)).map(tok, batched=True, remove_columns=["text", "summary"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    strategy_kwargs = training_strategy_kwargs(Seq2SeqTrainingArguments)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        save_strategy="epoch",
        **strategy_kwargs,
        **precision_kwargs(deterministic),
        learning_rate=lr,
        per_device_train_batch_size=max(1, batch_size // 2),
        per_device_eval_batch_size=max(1, batch_size // 2),
        num_train_epochs=epochs,
        **training_seed_kwargs(Seq2SeqTrainingArguments, seed),
        predict_with_generate=True,
        save_total_limit=1,
        dataloader_num_workers=0 if deterministic else 2,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    write_json(Path(output_dir) / "test_metrics.json", metrics)
    cleanup_checkpoints(output_dir, keep_checkpoints)
    return metrics


def training_strategy_kwargs(cls) -> dict:
    params = inspect.signature(cls).parameters
    if "eval_strategy" in params:
        return {"eval_strategy": "epoch"}
    return {"evaluation_strategy": "epoch"}


def training_seed_kwargs(cls, seed: int) -> dict:
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "seed" in params:
        kwargs["seed"] = seed
    if "data_seed" in params:
        kwargs["data_seed"] = seed
    return kwargs


def cleanup_checkpoints(output_dir: str | Path, keep_checkpoints: bool) -> None:
    if keep_checkpoints:
        return
    root = Path(output_dir)
    for path in root.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)
