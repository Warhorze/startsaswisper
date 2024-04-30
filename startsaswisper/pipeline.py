import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import wer, mer, wil
from icecream import ic


MODEL_ID :str = "distil-whisper/distil-medium.en"
DATASET_ID  : str = "hf-internal-testing/librispeech_asr_dummy"
MAX_TOKENS : int = 128
SAMPLES : int = 10

def main()-> None:
    #  https://github.com/huggingface/distil-whisper
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = MODEL_ID

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


    dataset = load_dataset(DATASET_ID, "clean", split="validation")
    sample_range= range(SAMPLES)
    for i in sample_range:
        sample = dataset[i]["audio"]
        reference = dataset[i]['text'].lower()
        result = pipe(sample)['text']
        ic(result)
        ic(reference)
        wer_score = wer(result, reference)
        ic(wer_score)
        wil_score = wil(result, reference)
        ic(wil_score)

if __name__ == "__main__":
    main()


