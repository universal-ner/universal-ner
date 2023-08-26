import fire
import torch
from transformers import pipeline

from ..utils import preprocess_instance

def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    max_new_tokens: int = 256,
):    
    generator = pipeline('text-generation', model=model_path, torch_dtype=torch.float16, device=0)
    while True:
        try:
            text = input('Input text: ')
            entity_type = input('Input entity type: ')
        except EOFError:
            text = entity_type = ''
        if not text:
            print("Exit...")
            break
        example = {"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}
        prompt = preprocess_instance(example['conversations'])
        outputs = generator(prompt, max_length=max_new_tokens, return_full_text=False)
        print(outputs[0]['generated_text'])

if __name__ == "__main__":
    fire.Fire(main)