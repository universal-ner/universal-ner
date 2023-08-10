import fire
import gradio as gr
from transformers import LlamaTokenizer
from vllm import LLM

from .inference import inference

def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    tensor_parallel_size: int = 1,
    max_input_length: int = 512,
):    

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    def evaluate(
        text,
        entity_type,
        max_new_tokens=128,
    ):
        if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
            print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")
        examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
        output = inference(llm, examples, max_new_tokens=max_new_tokens)[0]
        yield output

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="Enter an input text."),
            gr.components.Textbox(lines=2, label="Entity type", placeholder="Enter an entity type."),
            gr.components.Slider(
                minimum=1, maximum=256, step=1, value=64, label="Max output tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        examples = [
            ['UniversalNER surpasses existing instruction-tuned models at the same size (e.g., Alpaca, Vicuna) by a large margin, and shows substantially better performance to ChatGPT.', 'model'],
            ['Raccoons and red pandas belong to different families: raccoons are part of the Procyonidae family, while red pandas belong to the Ailuridae family. Unlike raccoons, red pandas have a very restricted geographical range, while raccoons can be found across various regions worldwide.', 'animal']
        ],
        title="UniversalNER",
    ).queue().launch(server_name="0.0.0.0")

if __name__ == "__main__":
    fire.Fire(main)