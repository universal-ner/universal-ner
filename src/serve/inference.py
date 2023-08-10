from vllm import LLM, SamplingParams
from typing import List, Type

from ..utils import preprocess_instance, get_response

def inference(
    model: Type[LLM],
    examples: List[dict],
    max_new_tokens: int = 256,
):
    prompts = [preprocess_instance(example['conversations']) for example in examples]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])
    responses = model.generate(prompts, sampling_params)
    responses_corret_order = []
    response_set = {response.prompt: response for response in responses}
    for prompt in prompts:
        assert prompt in response_set
        responses_corret_order.append(response_set[prompt])
    responses = responses_corret_order
    outputs = get_response([output.outputs[0].text for output in responses])
    return outputs
