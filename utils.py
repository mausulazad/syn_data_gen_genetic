import requests
import copy
import torch
from PIL import Image
from typing import Dict

import ast
import sys
import warnings
import os

import hashlib

from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, GenerationConfig

from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# LLava-Critic
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from datasets import load_dataset
from models import MLLM, Judge, BackwardReasoner, FinalJudge

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
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    return (model, processor)

def setup_llava():
    # TODO: Later, upgrade to 1.6
    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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
            generator_mllms.append(MLLM(model, processor, model_family="llama_32", inference_type="generate"))
        elif model_name == "molmo":
            # NOT WORKING: Bug fix is needed
            model, processor = setup_molmo()
            generator_mllms.append(MLLM(model, processor, model_family="molmo", inference_type="generate"))
        elif model_name == "llava":
            model, processor = setup_llava()
            generator_mllms.append(MLLM(model, processor, model_family="llava", inference_type="generate"))    
    return generator_mllms

'''
def setup_llava_critic():
    warnings.filterwarnings("ignore")
    model_id = "lmms-lab/llava-critic-7b"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_id, 
        None, 
        model_name, 
        device_map=device_map
    ) 

    #url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    #mage = Image.open(requests.get(url, stream=True).raw)
    #image_tensor = process_images([image], image_processor, model.config)
    #image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

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
    
def setup_judge_models(judge_models):
    judge_mllms = []
    for model_name in judge_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            judge_mllms.append(Judge(model, processor, model_family="llama_32", inference_type='judge'))
        # TODO: Pr 3 --> add other judge mllms (i.e. llava-critic, blip-3, sfr judge, prometheus)
        elif model_name == "llava_critic":
            #setup_llava_critic()
            pass
        elif model_name == "phi_3_vision":
            model, processor = setup_phi3_vision()
            judge_mllms.append(Judge(model, processor, model_family="phi_3_vision", inference_type='judge'))
    return judge_mllms

def setup_backward_reasoning_models(br_models):
    br_mllms = []
    for model_name in br_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            br_mllms.append(BackwardReasoner(model, processor, model_family="llama_32", inference_type='backward_reasoning'))
        # TODO: Pr 3 --> add other judge mllms (i.e. llava-critic, blip-3, sfr judge, prometheus)
        elif model_name == "phi_3_vision":
            model, processor = setup_phi3_vision()
            br_mllms.append(BackwardReasoner(model, processor, model_family="phi_3_vision", inference_type='backward_reasoning'))
    return br_mllms

def setup_models(generator_models, judge_model, br_model):
    # Remove after testing
    generator_mllms = []
    judge_mllm = None
    br_mllm = None
    
    generator_mllms = setup_generator_models(generator_models)
    #judge_mllm = setup_judge_models([judge_model])[0]
    #br_mllm = setup_backward_reasoning_models([br_model])[0]
    
    return (generator_mllms, judge_mllm, br_mllm)

def deduplicate_qars(qars):
    unique_qars = []
    seen_qar_hashes = set()
    # step 1: remove exact copies of qars
    for qar in qars:
        key = f'{qar["question"]}, {qar["answer"]}, {qar["rationale"]}'
        qar_hash = hashlib.md5(key.encode()).hexdigest()
        if qar_hash not in seen_qar_hashes:
            unique_qars.append(qar)
            seen_qar_hashes.add(qar_hash)

    # step 2: remove highly similar qars (1 of them)
    nonsimilar_qars = []
    seen_qars = []
    for qar in unique_qars:
        qar_1_text = f'{qar["question"]}, {qar["answer"]}, {qar["rationale"]}'
        qar_is_duplicate = False
        for seen_qar in seen_qars:
            qar_2_text = f'{seen_qar["question"]}, {seen_qar["answer"]}, {seen_qar["rationale"]}'
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([qar_1_text, qar_2_text]).toarray()
            cosine_sim = cosine_similarity(vectors)[0, 1]
            qar_is_duplicate = cosine_sim >= 0.85
            if qar_is_duplicate:
                break

        if not qar_is_duplicate:
            seen_qars.append(qar)
            nonsimilar_qars.append(qar)

    return nonsimilar_qars

def setup_final_judge():
    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
    final_judge = FinalJudge(model, processor, tokenizer, max_length)
    return final_judge

def get_gpu_details():
    gpu_count = torch.cuda.device_count()
    print(f'No. of GPUs: {gpu_count}')
    for i in range(gpu_count):
        print(f"Device cuda:{i} - {torch.cuda.get_device_name(i)}")

# TODO
def estimate_model_ram_usage():
    pass



