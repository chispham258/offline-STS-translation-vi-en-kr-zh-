#!/usr/bin/env python3
"""Convert openai/whisper-large-v3-turbo to CTranslate2 INT8 format.

Usage:
    uv run python scripts/convert_whisper_to_ct2.py --quantization int8
    uv run python scripts/convert_whisper_to_ct2.py --quantization int8_float16
    uv run python scripts/convert_whisper_to_ct2.py --quantization float16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import ctranslate2
from transformers import WhisperProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Whisper to CTranslate2 format"
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3-turbo",
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--output",
        default="models/whisper-large-v3-turbo-ct2-int8",
        help="Output directory for CTranslate2 model",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="Quantization type",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} exists. Use --force to overwrite."
            )

    print(f"Converting {args.model} to CTranslate2 format...")
    print(f"Quantization: {args.quantization}")
    print(f"Output: {output_dir}")

    # Create converter
    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=args.model,
        load_as_float16=True,  # Load as FP16 first for faster conversion
    )

    # Convert
    converter.convert(
        output_dir=str(output_dir),
        quantization=args.quantization,
        force=args.force,
    )

    # Also save processor (tokenizer + feature extractor)
    print("Saving processor...")
    processor = WhisperProcessor.from_pretrained(args.model)
    processor.save_pretrained(output_dir)

    print("\n✅ Conversion complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Files: {list(output_dir.iterdir())}")


if __name__ == "__main__":
    main()