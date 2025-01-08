from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer 
import torch 
import nltk 
import os
from diffusers import DiffusionPipeline
from pathlib import Path

ROOT = Path(__file__).parent

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

SEED = torch.manual_seed(336) 

llm_args = {
    'max_length': 100,
    'no_repeat_ngram_size': 1,
    #'temperature': 1.3,
    'top_p': 0.9,
    'top_k': 100,
    'do_sample': True
}

image_params = {'num_inference_steps':50, 
                'num_images_per_prompt':1,
                'generator':SEED, 
                'guidance_scale':15,
                'negative_prompt':"animated, bad, terrible, low quality, weird"} 


# Get 3 sentences or lesser from the generated prompt
def get_processed_prompt(text):
    sentences = nltk.sent_tokenize(text)
    return ' '.join(sentences[:2])
    

old_gpt_model = AutoModelForCausalLM.from_pretrained(f'{ROOT}/models/fine_tuned_gpt') 
old_gpt_tokenizer = AutoTokenizer.from_pretrained(f'{ROOT}/models/fine_tuned_gpt') 

new_gpt_model = AutoModelForCausalLM.from_pretrained(f'{ROOT}/models/fine_tuned_gpt_new_data') 
new_gpt_tokenizer = AutoTokenizer.from_pretrained(f'{ROOT}/models/fine_tuned_gpt_new_data') 

t5model = T5ForConditionalGeneration.from_pretrained(f'{ROOT}/models/fine_tuned_t5') 
t5tokenizer = T5Tokenizer.from_pretrained(f'{ROOT}/models/fine_tuned_t5') 


# Function to generate images from enhanced prompts
def generate_images_and_prompts(input_text):


    # Original Image
    og_image = pipe.to(torch.device('cuda'))(input_text, **image_params).images[0]


    #T5 image
    # Tokenize the input text
    t5_inputs = t5tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    t5_outputs = t5model.to(torch.device('cpu')).generate(
        input_ids=t5_inputs['input_ids'],
        attention_mask=t5_inputs['attention_mask'],
        temperature=0.3,
        **llm_args
    )
    t5_generated_text = get_processed_prompt(t5tokenizer.decode(t5_outputs[0], skip_special_tokens=True))
    t5_image = pipe.to(torch.device('cuda'))(t5_generated_text, **image_params).images[0]


    # GPT image
    old_gpt_inputs = old_gpt_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    outputs = old_gpt_model.to(torch.device('cpu')).generate(
        input_ids=old_gpt_inputs['input_ids'],
        attention_mask=old_gpt_inputs['attention_mask'],
        temperature=0.3,
        **llm_args
    )
    gpt_generated_text = get_processed_prompt(old_gpt_tokenizer.decode(outputs[0], skip_special_tokens=True))
    gpt_image = pipe.to(torch.device('cuda'))(gpt_generated_text, **image_params).images[0]


    # New GPT image
    new_gpt_inputs = new_gpt_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    new_gpt_outputs = new_gpt_model.to(torch.device('cpu')).generate(
        input_ids=new_gpt_inputs['input_ids'],
        attention_mask=new_gpt_inputs['attention_mask'],
        temperature=1.5,
        **llm_args
    )
    new_gpt_generated_text = get_processed_prompt(new_gpt_tokenizer.decode(new_gpt_outputs[0], skip_special_tokens=True))
    new_gpt_image = pipe.to(torch.device('cuda'))(new_gpt_generated_text, **image_params).images[0]

 
    return input_text, og_image, t5_generated_text, t5_image, gpt_generated_text, gpt_image, new_gpt_generated_text, new_gpt_image


import gradio as gr

# Gradio Interface setup
iface = gr.Interface(
    fn=generate_images_and_prompts, 
    inputs=gr.Textbox(label="Enter your prompt for image generation"),
    outputs=[
        gr.Textbox(label="Original Prompt"),
        gr.Image(type="pil", label="Image generated from Original Prompt"),

        gr.Textbox(label="Prompt enhanced by T5"),
        gr.Image(type="pil", label="Generated Image"),

        gr.Textbox(label="Prompt enhanced by GPT on old data"),
        gr.Image(type="pil", label="Generated Image"),

        gr.Textbox(label="Prompt enhanced by GPT on new data"),
        gr.Image(type="pil", label="Generated Image"),

    ],
    live=False,
    description="App"
)

# Launch the Gradio app
iface.launch()


