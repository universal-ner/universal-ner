from .conversation import get_conv_template

def preprocess_instance(source):
    conv = get_conv_template("ie_as_qa")
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1:
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt

def get_response(responses):
    responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
    return responses
