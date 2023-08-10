import fire
from vllm import LLM
from transformers import LlamaTokenizer

from .inference import inference

def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    max_new_tokens: int = 256,
    tensor_parallel_size: int = 1,
    max_input_length: int = 512,
):    

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    while True:
        try:
            text = input('Input text: ')
            entity_type = input('Input entity type: ')
        except EOFError:
            text = entity_type = ''
        if not text:
            print("Exit...")
            break
        if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
            print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")
            continue
        examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
        output = inference(llm, examples, max_new_tokens=max_new_tokens)[0]
        print(output)

if __name__ == "__main__":
    fire.Fire(main)