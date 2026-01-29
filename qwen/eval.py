import argparse
import csv
import json
import random
from dataclasses import dataclass
from typing import List

import torch
from peft import PeftModel
from PIL import Image
import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import evaluate


@dataclass
class EvalExample:
    image: str
    references: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL LoRA vs base model on COCO-style captions."
    )
    parser.add_argument(
        "--base-model",
        default="./Qwen/Qwen2.5-VL-7B-Instruct",
        help="Path or HF hub id for the base model.",
    )
    parser.add_argument(
        "--lora-path",
        default="./output/Qwen2_5-VL-7B/checkpoint-930",
        help="Path to LoRA checkpoint to evaluate.",
    )
    parser.add_argument(
        "--val-csv",
        default="val2014.csv",
        help="Validation CSV with captions and image paths.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate per sample.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--output",
        default="eval_metrics.json",
        help="Path to write evaluation metrics.",
    )
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    return Image.open(image_path).convert("RGB")


def load_eval_examples(val_csv: str, max_samples: int, seed: int) -> List[EvalExample]:
    examples: List[EvalExample] = []
    with open(val_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            captions = [caption.strip() for caption in row["caption"].split("&&") if caption.strip()]
            examples.append(EvalExample(image=row["image"], references=captions))

    random.Random(seed).shuffle(examples)
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def safe_load_metric(name: str):
    try:
        return evaluate.load(name)
    except (FileNotFoundError, ModuleNotFoundError, ValueError):
        return None


def build_metrics():
    metrics = {
        "sacrebleu": safe_load_metric("sacrebleu"),
        "rouge": safe_load_metric("rouge"),
        "meteor": safe_load_metric("meteor"),
        "cider": safe_load_metric("cider"),
    }
    missing = [key for key, value in metrics.items() if value is None]
    if missing:
        print(
            "[warn] Metrics not available: "
            + ", ".join(missing)
            + ". Install with: pip install evaluate sacrebleu rouge-score nltk"
        )
        if "meteor" in missing:
            print("[warn] METEOR needs NLTK data: python -m nltk.downloader wordnet")
    return metrics


def generate_caption(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def evaluate_model(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    examples: List[EvalExample],
    max_new_tokens: int,
) -> dict:
    metrics = build_metrics()
    predictions: List[str] = []
    references: List[List[str]] = []

    for example in examples:
        image = load_image(example.image)
        prediction = generate_caption(model, processor, image, max_new_tokens)
        predictions.append(prediction)
        references.append(example.references)

    results = {"num_samples": len(predictions)}
    if metrics["sacrebleu"]:
        results["sacrebleu"] = metrics["sacrebleu"].compute(
            predictions=predictions, references=references
        )
    if metrics["rouge"]:
        results["rouge"] = metrics["rouge"].compute(predictions=predictions, references=references)
    if metrics["meteor"]:
        results["meteor"] = metrics["meteor"].compute(predictions=predictions, references=references)
    if metrics["cider"]:
        results["cider"] = metrics["cider"].compute(predictions=predictions, references=references)

    return results


def main() -> None:
    args = parse_args()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    examples = load_eval_examples(args.val_csv, args.max_samples, args.seed)

    processor = AutoProcessor.from_pretrained(args.base_model)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    base_results = evaluate_model(base_model, processor, examples, args.max_new_tokens)

    del base_model
    torch.cuda.empty_cache()

    lora_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(lora_base, model_id=args.lora_path)
    lora_results = evaluate_model(lora_model, processor, examples, args.max_new_tokens)

    output = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "metrics": {
            "base": base_results,
            "lora": lora_results,
        },
    }

    with open(args.output, "w", encoding="utf-8") as outfile:
        json.dump(output, outfile, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
