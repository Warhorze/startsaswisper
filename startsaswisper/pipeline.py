import argparse
from typing import Dict, List

import jiwer
import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID: str = "openai/whisper-small"
METADATA_PATH = "api/recordings.csv"
MAX_TOKENS: int = 128
SAMPLES: int = 20


def filter_data(
    df: pd.DataFrame,
    created_from_date: str = None,
    created_to_date: str = None,
    user_id: int = None,
    unit_id: int = None,
) -> List[str]:
    """Filter the dataset based on the provided criteria and return the indecises of the selected recordings"""
    df["created_at"] = pd.to_datetime(df["created_at"])

    conditions = (
        (
            (df["created_at"] >= created_from_date)
            if created_from_date is not None
            else True
        ),
        (
            (df["created_at"] <= created_to_date)
            if created_to_date is not None
            else True
        ),
        (df["user_id"] == user_id) if user_id is not None else True,
        (df["unit_id"] == unit_id) if unit_id is not None else True,
    )

    # Apply filters
    filtered_df = df[
        (conditions[0]) & (conditions[1]) & (conditions[2]) & (conditions[3])
    ]

    return filtered_df.index.to_list()


def get_dataset(sampling_rate: int = 1600, samples: int = 20):
    common_voice = load_dataset(
        "mozilla-foundation/common_voice_11_0", "nl", split="test"
    )
    common_voice = common_voice.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return common_voice[:samples]


def evaluate(dataset: List[tuple], pipeline, scoring_metric=jiwer.wer):
    """transcribe and evaluate a dataset"""
    results = []
    for audio, reference in dataset:
        result = pipeline(audio, generate_kwargs={"language": "dutch"})["text"]
        score = scoring_metric(result, reference)
        results.append(
            {
                "transcript": result,
                "original": reference,
                "score": score,
                "metric": scoring_metric.__name__,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model_id",
        type=str,
        choices=["whisper-small", "whisper-tiny"],
        help="The ID of the model",
    )
    args = parser.parse_args()

    try:
        if args.model_id:
            print("Model ID:", args.model_id)
        else:
            print("Please provide a valid model ID using the --model_id flag.")
    except argparse.ArgumentError:
        print(
            "Invalid model ID. Available options are 'whisper-small' and 'whisper-tiny'."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=MAX_TOKENS,
        torch_dtype=torch_dtype,
        device=device,
    )

    dataset = get_dataset(samples=SAMPLES)
    pdf = pd.read_csv(METADATA_PATH)
    indx_records = filter_data(pdf, unit_id=2)

    selected_records = [
        (dataset["audio"][i], dataset["sentence"][i].lower()) for i in indx_records
    ]
    results = evaluate(selected_records, pipeline=pipe)
    print(results)

    return results


if __name__ == "__main__":
    main()
