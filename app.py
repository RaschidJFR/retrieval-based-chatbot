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


def predict(message, chatbot):
    
    print(f"Logging: message is - {message}")
    print(f"Logging: chatbot is - {chatbot}")

    input_prompt = f"[INST]<<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in chatbot:
        input_prompt = input_prompt + interaction[0] + " [/INST] " + interaction[1] + " </s><s> [INST] "

    input_prompt = input_prompt + message + " [/INST] "

    print(f"Logging: input_prompt is - {input_prompt}")
    data = {
        "inputs": input_prompt,
        "parameters": {"max_new_tokens":256}
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data), auth=('hf', hf_token))

    print(f'Logging: API response is - {response.text}')
    response_json_object = json.loads(response.text)
    return response_json_object[0]['generated_text']


gr.ChatInterface(predict, title=title, description=description).queue().launch(debug= True) 
