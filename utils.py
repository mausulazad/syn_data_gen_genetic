import requests
import copy
import torch
from PIL import Image
from typing import Dict

import ast
import sys
import warnings
import os

from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, GenerationConfig

#from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# LLava-Critic
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from datasets import load_dataset
from models import MLLM, Judge, BackwardReasoner

def load_and_preprocess_dataset(dataset_name):
    if dataset_name == "aokvqa":
        dataset = load_dataset("HuggingFaceM4/A-OKVQA")
        # We are generating synthetic qar by using training data as seed dataset
        # Later we will finetune using test data (and validation data as needed)
        return dataset["train"]

def setup_llama32():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return (model, processor)

def setup_molmo():
    model_id = "allenai/Molmo-7B-D-0924"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    return (model, processor)

def setup_generator_models(generator_models):
    generator_mllms = []
    for model_name in generator_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            generator_mllms.append(MLLM(model, processor, model_family="llama_32", inference_type='generate'))
        elif model_name == "molmo":
            # NOT WORKING: Bug fix is needed
            model, processor = setup_molmo()
            generator_mllms.append(MLLM(model, processor, model_family="molmo", inference_type='generate'))
    return generator_mllms

def setup_llava_critic():
    '''
    # Install 'llava' library' prior to loading LLaVa-Critci: pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    model_id = "lmms-lab/llava-critic-7b"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    '''

    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "cuda"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) 

    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=300,
    )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)
    
    '''
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_id, 
        None, 
        model_name, 
        device_map="auto"
    )
    '''

def setup_phi3_vision():
    model_id = "microsoft/Phi-3.5-vision-instruct" 

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2'
    )

    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        num_crops=16
    )

    return (model, processor)

    '''
    

    

    
    
    score, feedback = ast.literal_eval(response)
    '''
    
def setup_judge_models(judge_models):
    judge_mllms = []
    for model_name in judge_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            judge_mllms.append(Judge(model, processor, model_family="llama_32", inference_type='judge'))
        # TODO: Pr 3 --> add other judge mllms (i.e. llava-critic, blip-3, sfr judge, prometheus)
        elif model_name == "llava_critic":
            setup_llava_critic()
        elif model_name == "phi_3_vision":
            model, processor = setup_phi3_vision()
            judge_mllms.append(Judge(model, processor, model_family="phi_3_vision", inference_type='judge'))
    return judge_mllms

def setup_backward_reasoning_models(br_models):
    br_mllms = []
    for model_name in br_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            br_mllms.append(BackwardReasoner(model, processor, inference_type='backward_reasoning'))
        # TODO: Pr 3 --> add other judge mllms (i.e. llava-critic, blip-3, sfr judge, prometheus)
    return br_mllms

def setup_models(generator_models, judge_model, br_model):
    generator_mllms = []
    #generator_mllms = setup_generator_models(generator_models)
    #judge_mllm = setup_judge_models([judge_model])[0]
    judge_mllm = setup_judge_models([judge_model])[0]
    #br_mllm = setup_backward_reasoning_models([br_model])[0]
    
    #return (generator_mllms, judge_mllm, br_mllm)
    return (generator_mllms, judge_mllm)
    #return (generator_mllms) 