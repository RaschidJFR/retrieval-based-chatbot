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
description = """This Space demonstrates model [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta, running on Inference Endpoints using text-generation-inference. To have your own dedicated endpoint, you can [deploy it on Inference Endpoints](https://ui.endpoints.huggingface.co/). """ 
css = """.toast-wrap { display: none !important } """

def predict(message, chatbot):
    
    input_prompt = f"[INST]<<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in chatbot:
        input_prompt = input_prompt + interaction[0] + " [/INST] " + interaction[1] + " </s><s> [INST] "

    input_prompt = input_prompt + message + " [/INST] "

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
                gr.Warning("This line does not start with 'data:':", decoded_line)
                continue

            # Load as JSON
            try:
                partial_message = partial_message + json.loads(json_line)['token']['text'] 
                yield partial_message
            except json.JSONDecodeError:
                gr.Warning("This line is not valid JSON: ", json_line)
                continue

gr.ChatInterface(predict, title=title, description=description, css=css).queue(concurrency_count=75).launch() 
