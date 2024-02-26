import gradio as gr
import os
from huggingface_hub import AsyncInferenceClient
from pdf import get_documentation_text

HF_TOKEN = os.getenv('HF_TOKEN')
api_url = os.getenv('API_URL')
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
client = AsyncInferenceClient(api_url)

system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
chatbot_instructions = "Read this document. Any future prompt I ask you will be related to this content. \n\n" + get_documentation_text()
title = "Llama2 70B Chatbot"
description = """
This Space demonstrates model [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta, a Llama 2 model with 70B parameters fine-tuned for chat instructions. This space is running on Inference Endpoints using text-generation-inference library. If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://ui.endpoints.huggingface.co/).

🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at our blog post](https://huggingface.co/blog/llama2).

🔨 Looking for lighter chat model versions of Llama-v2? 
- 🐇 Check out the [7B Chat model demo](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat).
- 🦊 Check out the [13B Chat model demo](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat).

Note: As a derivate work of [Llama-2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/USE_POLICY.md).
"""
css = """.toast-wrap { display: none !important } """


# Note: We have removed default system prompt as requested by the paper authors [Dated: 13/Oct/2023]
# Prompting style for Llama2 without using system prompt
# <s>[INST] {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]


# Stream text - stream tokens with InferenceClient from TGI
async def predict(message, chatbot, system_prompt="", temperature=0.9, max_new_tokens=256, top_p=0.6, repetition_penalty=1.0,):
    
    # Initialize the input prompt with initial instructions
    input_prompt = f"<s>[INST] {chatbot_instructions} [/INST] Ok </s><s>[INST] "
    
    if system_prompt != "":
        input_prompt = input_prompt + f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n "
    else:
        input_prompt = input_prompt + f"<s>[INST] "
        
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    
    for interaction in chatbot:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s>[INST] "

    input_prompt = input_prompt + str(message) + " [/INST] "

    partial_message = ""
    async for token in await client.text_generation(prompt=input_prompt, 
                                    max_new_tokens=max_new_tokens, 
                                    stream=True, 
                                    best_of=1, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    do_sample=True, 
                                    repetition_penalty=repetition_penalty):
        partial_message = partial_message + token 
        yield partial_message
        


additional_inputs=[
    gr.Textbox("", label="Optional system prompt"),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=4096,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.6,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

chatbot_stream = gr.Chatbot(avatar_images=('user.png', 'bot2.png'),bubble_full_width = False)
chat_interface_stream = gr.ChatInterface(predict, 
                 title=title, 
                 description=description, 
                 textbox=gr.Textbox(),
                 chatbot=chatbot_stream,
                 css=css, 
                 #cache_examples=True, 
                 additional_inputs=additional_inputs,) 

# Gradio Demo 
with gr.Blocks() as demo:

    with gr.Tab("Streaming"):
        # streaming chatbot
        #chatbot_stream.like(vote, None, None)
        chat_interface_stream.render()

demo.queue(max_size=100).launch()