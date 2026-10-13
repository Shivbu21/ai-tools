#!/usr/bin/env python3
"""Fine-tune Whisper on code-switched Hindi data (transformers Seq2SeqTrainer).

!!! REQUIRES A CUDA GPU !!!
A full fine-tune on CPU is impractical (hours per epoch even for whisper-tiny
on a few hundred clips). This script refuses to train on CPU unless you pass
--force-cpu, and offers --smoke-test to validate the whole pipeline wiring
(data loading, collator, forward+backward pass) with a single step on CPU.

Pipeline: train.jsonl/dev.jsonl from prep_data.py -> WhisperProcessor ->
DataCollatorSpeechSeq2SeqWithPadding -> Seq2SeqTrainer -> ./whisper-hinglish/

Usage (on a GPU box):
    pip install torch transformers datasets accelerate soundfile librosa
    python prep_data.py --cv-samples 2000
    python finetune.py --model openai/whisper-tiny --epochs 3

Smoke test (validates code on this CPU box, trains nothing):
    python finetune.py --smoke-test
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--model", default="openai/whisper-tiny",
                    help="HF hub id or local whisper checkpoint")
    ap.add_argument("--output-dir",
                    default=os.path.join(HERE, "whisper-hinglish"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--language", default="Hindi",
                    help="Forced decoder language token")
    ap.add_argument("--smoke-test", action="store_true",
                    help="One training step on CPU to validate the pipeline")
    ap.add_argument("--force-cpu", action="store_true",
                    help="Allow real training on CPU (slow; not recommended)")
    args = ap.parse_args()

    import torch
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor,
                              Seq2SeqTrainingArguments, Seq2SeqTrainer)
    from transformers.models.whisper.modeling_whisper import shift_tokens_right
    from datasets import Dataset

    use_cuda = torch.cuda.is_available()
    if args.smoke_test:
        print("SMOKE TEST: single training step on CPU (validates wiring).")
    elif not use_cuda and not args.force_cpu:
        print("ERROR: no CUDA GPU detected. Fine-tuning Whisper on CPU is "
              "impractical.", file=sys.stderr)
        print("Run on a GPU machine, or use --smoke-test to validate the "
              "pipeline wiring with one CPU step.", file=sys.stderr)
        return 2

    train_rows = load_manifest(os.path.join(args.data_dir, "train.jsonl"))
    dev_rows = load_manifest(os.path.join(args.data_dir, "dev.jsonl"))
    print(f"train: {len(train_rows)} utterances, dev: {len(dev_rows)}")

    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe")

    def prepare(batch):
        import librosa
        import soundfile as sf
        audios, texts = [], []
        for wav_path, text in zip(batch["audio"], batch["text"]):
            audio, sr = sf.read(wav_path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audios.append(audio)
            texts.append(text)
        feats = processor.feature_extractor(
            audios, sampling_rate=16000).input_features
        labels = processor.tokenizer(
            texts, padding=True).input_ids
        return {"input_features": feats, "labels": labels}

    def to_hf(rows):
        return Dataset.from_dict(
            {"audio": [r["audio"] for r in rows],
             "text": [r["text"] for r in rows]})

    train_ds = to_hf(train_rows).map(
        prepare, batched=True, batch_size=16,
        remove_columns=["audio", "text"])
    dev_ds = to_hf(dev_rows).map(
        prepare, batched=True, batch_size=16,
        remove_columns=["audio", "text"])

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "hi"
    model.generation_config.task = "transcribe"

    from dataclasses import dataclass

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: object

        def __call__(self, features):
            import torch
            input_features = [{"input_features": f["input_features"]}
                              for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs if not args.smoke_test else 1,
        max_steps=1 if args.smoke_test else -1,
        fp16=use_cuda,
        gradient_checkpointing=True,
        eval_strategy="epoch" if not args.smoke_test else "no",
        save_strategy="epoch" if not args.smoke_test else "no",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    import inspect
    # transformers>=5 renamed Trainer's `tokenizer` kwarg -> `processing_class`
    if "processing_class" in inspect.signature(
            Seq2SeqTrainer.__init__).parameters:
        proc_kwarg = {"processing_class": processor}
    else:  # transformers 4.x
        proc_kwarg = {"tokenizer": processor.feature_extractor}

    trainer = Seq2SeqTrainer(
        args=targs,
        model=model,
        train_dataset=train_ds,
        eval_dataset=None if args.smoke_test else dev_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        **proc_kwarg,
    )
    trainer.train()
    if not args.smoke_test:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"saved fine-tuned model to {args.output_dir}")
        print("Next: python eval_finetune.py --model-dir", args.output_dir)
    else:
        print("SMOKE TEST PASSED: data -> collator -> forward+backward OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
