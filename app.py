import json 
import gradio as gr
import os
import requests

hf_token = os.getenv('HF_TOKEN')
api_url = os.getenv('API_URL') 
headers = {
    'Content-Type': 'application/json',
}

system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
title = "Llama2 70B Chatbot"
#description = """This Space demonstrates model [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta, running on Inference Endpoints using text-generation-inference. To have your own dedicated endpoint, you can [deploy it on Inference Endpoints](https://ui.endpoints.huggingface.co/). """ 
description = """
This Space demonstrates model [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta, a Llama 2 model with 70B parameters fine-tuned for chat instructions. This space is running on Inference Endpoints using text-generation-inference library. If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://ui.endpoints.huggingface.co/).

🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at our blog post](https://huggingface.co/blog/llama2).

🔨 Looking for a lighter chat model version of Llama-v2? Check out the large [7B Chat model demo](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat).

Note: As a derivate work of [Llama-2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/USE_POLICY.md).
"""
css = """.toast-wrap { display: none !important } """
examples=[
    'Hello there! How are you doing?',
    'Can you explain to me briefly what is Python programming language?',
    'Explain the plot of Cinderella in a sentence.',
    'How many hours does it take a man to eat a Helicopter?',
    "Write a 100-word article on 'Benefits of Open-Source in AI research'",
    ]


def predict(message, chatbot):
    
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in chatbot:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "

    input_prompt = input_prompt + str(message) + " [/INST] "

    data = {
        "inputs": input_prompt,
        "parameters": {"max_new_tokens":256}
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data), auth=('hf', hf_token), stream=True)
    
    partial_message = ""
    for line in response.iter_lines():
        if line:  # filter out keep-alive new lines
            # Decode from bytes to string
            decoded_line = line.decode('utf-8')

            # Remove 'data:' prefix 
            if decoded_line.startswith('data:'):
                json_line = decoded_line[5:]  # Exclude the first 5 characters ('data:')
            else:
                gr.Warning(f"This line does not start with 'data:': {decoded_line}")
                continue

            # Load as JSON
            try:
                json_obj = json.loads(json_line)
                if 'token' in json_obj:
                    partial_message = partial_message + json_obj['token']['text'] 
                    yield partial_message
                elif 'error' in json_obj:
                    yield json_obj['error'] + '. Please refresh and try again with an appropriate smaller input prompt.'
                else:
                    gr.Warning(f"The key 'token' does not exist in this JSON object: {json_obj}")

            except json.JSONDecodeError:
                gr.Warning(f"This line is not valid JSON: {json_line}")
                continue
            except KeyError as e:
                gr.Warning(f"KeyError: {e} occurred for JSON object: {json_obj}")
                continue

gr.ChatInterface(predict, title=title, description=description, css=css, examples=examples, cache_examples=True).queue(concurrency_count=75).launch() 
