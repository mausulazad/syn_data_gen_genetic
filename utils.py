import requests
import pprint
import copy
import random
import json
import torch
from PIL import Image
from typing import Dict
import io

import ast
import sys
import warnings
import os
import re

import hashlib
import secrets

from transformers import pipeline
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, GenerationConfig

#from fuzzywuzzy import fuzz
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from huggingface_hub import HfApi

#from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# LLava-Critic
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
# from llava.utils import disable_torch_init

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from datasets import load_dataset, Dataset
from models import MLLM, Judge, BackwardReasoner, FinalJudge

# pwd = os.getcwd()  
# os.environ['HF_HOME']=f'{pwd}/.cache'
# cache_dir=f'{pwd}/.cache'

cache_dir= '/scratch/mi8uu/cache'
os.environ['TRANSFORMERS_CACHE']=cache_dir
os.environ['HF_HOME']= cache_dir


CRITERIA = [
    "The question should require commonsense knowledge about human social behavior to answer.",
    "The question should require knowledge of the physical world to answer.",
    "The question should necessitate visual understanding to answer.",
    "The question should challenge the system's reasoning capabilities.",
    "The question should be sufficiently complex to require in-depth reasoning."
]

FEW_SHOT_QUESTIONS = [
    "What might happen if the person in the image dropped the object they are holding?",
    "How might the people in the image respond if one of them started laughing?",
    "Based on the setting in the image, what time of day is most likely depicted?",
    "What might the person in the image do next based on their posture?",
    "How could the objects in the image interact with one another?"
]

def load_and_preprocess_dataset(dataset_name):
    if dataset_name == "aokvqa":
        #dataset = load_dataset("HuggingFaceM4/A-OKVQA", cache_dir=cache_dir)
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="train", cache_dir=cache_dir)
        # We are generating synthetic qar by using training data as seed dataset
        # Later we will finetune using test data (and validation data as needed)
    else:
        dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    #return dataset["train"]
    return dataset

def setup_llama32(model_id, device):
    # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        #device_map="auto",
        #device_map=device,
        device_map={"": device},
        cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        #device_map='auto',
        #device_map=device,
        device_map={"": device},
        cache_dir=cache_dir
    )

    return (model, processor)

def setup_molmo(model_id, device):
    #model_id = "allenai/Molmo-7B-D-0924"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        #device_map="auto",
        device_map={"": device},
        cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        #device_map='auto',
        device_map={"": device},
        cache_dir=cache_dir
    )

    return (model, processor)

def setup_llava(model_id, device):
    # TODO: Later, upgrade to 1.6
    #model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        #device_map="auto",
        device_map={"": device},
        cache_dir=cache_dir
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        #device_map='auto',
        device_map={"": device},
        cache_dir=cache_dir
    )

    return (model, processor)

def setup_llava_critic(model_id, device):
    warnings.filterwarnings("ignore")
    model_id = "lmms-lab/llava-critic-7b"
    model_name = "llava_qwen"
    device_map="cuda"
    #device_map = device
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_id, 
        None, 
        model_name, 
        device_map=device_map,
        cache_dir=cache_dir
    )
    model.to(device)

    final_judge = FinalJudge("llava_critic", model, "qwen_1_5", image_processor, tokenizer, max_length) 
    return final_judge

def setup_prometheus_vision(model_id, device):
    #model_id = "kaist-ai/prometheus-vision-13b-v1.0"
    model_name = "llava-v1.5"
    #device_map = "auto"
    device_map = "cuda"
    #device_map = device
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id, 
        None, 
        model_name,
        device_map=device_map,
        cache_dir=cache_dir
    )
    #model = model.to("cuda")
    #model = model.to(device_map)
    model = model.to(device)

    final_judge = FinalJudge("prometheus_vision", model, "llava_v1", image_processor, tokenizer, max_length=context_len)
    return final_judge

def setup_qwen2_vl(model_id, device):
    #model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        #device_map="auto",
        #device_map=device,
        device_map={"": device},
        cache_dir=cache_dir
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        #device_map='auto',
        #device_map=device,
        device_map={"": device},
        cache_dir=cache_dir
    )
    
    final_judge = FinalJudge("qwen2_vl", model, None, processor)
    return final_judge

JURY_POLL = {
    "llava_critic": setup_llava_critic,
    "prometheus_vision": setup_prometheus_vision,
    "qwen2_vl": setup_qwen2_vl
}

def setup_jury_poll(jury_model_names, model_card):
    juries = []
    model_details = [model_config for model_config in model_card if model_config["name"] in jury_model_names]
    for model_entry in model_details:
        name = model_entry.get("name") 
        model_id = model_entry.get("model_id")
        device = model_entry.get("device")
        juries.append(JURY_POLL[name](model_id, device))
        #juries.append(JURY_POLL[jury_model_name](model_id, device))
    return juries

def setup_synthesizer_llm(model_name, model_card):
    model_details = [model_config for model_config in model_card if model_config["name"] == model_name]
    name = model_details[0].get("name") 
    model_id = model_details[0].get("model_id")
    device = model_details[0].get("device")
    if name == "qwen_25":
        #model_id= "Qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            #device_map="auto",
            #device_map=device,
            device_map={"": device},
            cache_dir=cache_dir
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir
        )

    return (model, tokenizer)

def synthesize_evol_methods(llm, question, evol_methods):
    system_prompt = """You are an helpful assistant with expertise in synthesizing text. You will be provided with **one to three evolution methods** generated by different Vision Language Models (VLMs) to evolve a question (question will be given too). Your task is to **identify the preferred method** among the provided options based on its alignment with the evaluation criteria and quality of suggestions. Then, synthesize a new evolution method by combining the best aspects of all provided methods, giving more weight to the preferred method. Ensure that the focus is  only on generating the synthesized evolution method and NOT on including or attempting to generate the evolved question itself.
        ### Evolution Criteria:
        1. Commonsense knowledge about human social behavior.
        2. Knowledge of the physical world.
        3. Visual understanding.
        4. Reasoning capabilities.
        5. Complexity requiring in-depth reasoning.        

        ### Step 1: Identify the Preferred Method
        Analyze the provided evolution methods and determine which one best satisfies the following:
        1. **Alignment with Evaluation Criteria**: Does the method directly address and improve the question's performance across all (or more than other evolution methods) evaluation criteria?
        2. **Clarity and Actionability**: Is the method easy to understand and implement without introducing ambiguity?
        3. **Balance**: Does the method enhance underperforming aspects of the question without compromising its strengths?

        ### Step 2: Synthesize the Evolution Method
        1. **For a Single Method**:
        - Use the single method as the final synthesized evolution method with no further modification unless explicitly required.

        2. **For Multiple Methods**:
        - Use the preferred method as the core framework for synthesis.
        - Extract and incorporate complementary suggestions from other methods only if they:
          - Address gaps or weaknesses in the preferred method without contradicting its core structure or logic.
          - Enhance the quality of the evolved question without introducing unnecessary complexity.
          - Align with the evaluation criteria.
        - During synthesis, explicitly identify and resolve any contradictions between the preferred method and other suggestions to ensure coherence and consistency in the final output.

        3. **Avoid Compromise**:
        - Ensure the synthesized method does not lower the quality of any evaluation criteria addressed in the preferred method.

        4. **Ensure Cohesion**:
        - Combine the suggestions seamlessly into a single coherent and logically flowing evolution method.

        ***IMPORTANT: The synthesized evolution method must ensure that the evolved question is designed to be concise and answerable using a maximum of 2-3 words, maintaining its alignment with the evaluation criteria.
        
        ### Output Format:
        Provide the results in the following structure:

        - **Preferred Method Identification**:  
        Indicate which method was chosen as the preferred one and explain why it was selected.

        - **Synthesis Process**:  
        Briefly explain how the preferred method was prioritized and how the other methods (if any) were used to enhance it.

        - **Synthesized Evolution Method**:
        Present the final synthesized evolution method as a clear, brief, concise, and actionable paragraph. Ensure the evolution method is focused on improving the question and does not include the evolved question itself.
        
        Make sure that '### Synthesized Evolution Method:' is always placed prior to stating the method."""
    
    model, tokenizer = llm
    
    # Shuffle evol method order to mitigate any bias
    random.shuffle(evol_methods)
    evol_methods = "\n".join(f"Evolution Method {i + 1}: {evol_method}" for i, evol_method in enumerate(evol_methods))
    query = f'Here is the question: {question}.\n And here are the evolution methods:\n{evol_methods}\n'
    
    messages = [
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": query }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    tries = 1
    final_evol_method = None
    while tries <= 3:
        generated_ids = model.generate(
            **model_inputs,
            temperature=0.3,
            max_new_tokens=350
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        synthesized_evol_method = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        splits = synthesized_evol_method.split("### Synthesized Evolution Method:")
        if len(splits) == 2:
            final_evol_method = splits[1].strip()
            break
        else:
            tries += 1
    return final_evol_method


def setup_phi3_vision():
    model_id = "microsoft/Phi-3.5-vision-instruct" 

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        #device_map="cuda",
        #device_map="auto", 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2',
        cache_dir=cache_dir
    )

    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        num_crops=16,
        cache_dir=cache_dir
    )
    return (model, processor)


def setup_llava_next(model_id, device):
    #model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        #device_map="cuda",
        #device_map="auto",
        #device_map=device,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype="auto",
        cache_dir=cache_dir
    )

    processor = LlavaNextProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        #device_map='auto',
        device_map={"": device},
        cache_dir=cache_dir
    )

    return (model, processor)

def setup_generator_models(generator_model_names, model_card):
    generator_mllms = []
    model_details = [model_config for model_config in model_card if model_config["name"] in generator_model_names]
    for model_entry in model_details:
        name = model_entry["name"] 
        model_id = model_entry["model_id"]
        device = model_entry["device"]
        if name == "llama_32":
            model, processor = setup_llama32(model_id, device)
            generator_mllms.append(MLLM(model, processor, model_family="llama_32", inference_type="generate"))
        elif name == "llava_next":
            model, processor = setup_llava_next(model_id, device)
            generator_mllms.append(MLLM(model, processor, model_family="llava_next", inference_type="generate"))
        elif name == "molmo":
            model, processor = setup_molmo(model_id, device)
            generator_mllms.append(MLLM(model, processor, model_family="molmo", inference_type="generate"))
        elif name == "llava":
            model, processor = setup_llava(model_id, device)
            generator_mllms.append(MLLM(model, processor, model_family="llava", inference_type="generate"))    
    return generator_mllms

    
"""
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
"""

"""
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
"""

"""
def setup_models(generator_models, judge_model, br_model):
    # Remove after testing
    generator_mllms = []
    judge_mllm = None
    br_mllm = None
    
    #generator_mllms = setup_generator_models(generator_models)
    #judge_mllm = setup_judge_models([judge_model])[0]
    #br_mllm = setup_backward_reasoning_models([br_model])[0]
    
    return (generator_mllms, judge_mllm, br_mllm)
"""


def setup_final_judge(model="llava_critic"):
    warnings.filterwarnings("ignore")
    '''
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
    final_judge = FinalJudge(model, processor, tokenizer, max_length)
    return final_judge
    '''

    if model == "llava_critic":
        print("Using LLaVA-Critic as final judge...")
        pretrained = "lmms-lab/llava-critic-7b"
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
        final_judge = FinalJudge(model, image_processor, tokenizer, max_length)
    elif model == "llava_next":
        print("Using LLaVA-Next as final judge...")
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            #device_map="cuda",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto"
        )

        processor = LlavaNextProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        final_judge = FinalJudge(model, processor)
    return final_judge


def get_gpu_details():
    gpu_count = torch.cuda.device_count()
    print(f'No. of GPUs: {gpu_count}')
    for i in range(gpu_count):
        print(f"Device cuda:{i} - {torch.cuda.get_device_name(i)}")

# TODO
def estimate_model_ram_usage():
    pass


def setup_slm(model_name, model_card):
    model = [model_details for model_details in model_card if model_details.get("name") == model_name]
    model = model[0]
    model_id = model["model_id"]
    device = model["device"]
    
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    slm = pipeline(
        "text-generation",
        model=model_id,
        max_new_tokens=1000,
        temperature=0.7,
        torch_dtype="auto",
        #device_map="auto",
        device_map=device
    )

    return slm

def postprocess_qars(slm, qar_text):
    system_prompt = """You are an helpful assistant who processes structured responses provided in a text format (such as QAR NO, QUESTION, ANSWER, RATIONALE) and converts them into a JSON-parseable list of triplets. Each triplet contains the fields:
        - "question": The question text.
        - "answer": The corresponding answer.
        - "rationale": The explanation or justification for the answer.

        ### Guidelines:
        1. Parse each response carefully to extract all valid question-answer-rationale (QAR) triplets.
        2. Return the results as a single JSON-parseable list of objects, where each object has the fields:
          {
            "question": <question text>,
            "answer": <corresponding correct answer>,
            "rationale": <corresponding rationale>
          }
        3. The input structure may differ from the examples provided (e.g., different formatting, separators, or label styles). Be flexible and adapt to extract QAR triplets from alternative structures while ensuring accuracy.
        4. Ensure the output strictly adheres to JSON format and contains only the list of objects. Do not include any additional text, explanation, or code in the response.
        5. If no valid QAR triplets are found in the input (e.g., the input contains no questions, answers, or rationales), return an empty JSON list: `[]`. Do not infer, hallucinate, or make up QAR triplets.
        6. If the input contains general statements, acknowledgments, or explanations unrelated to valid QAR triplets, do not attempt to interpret these as QARs. Return an empty list: `[]`.
        7. If a QAR triplet is incomplete or missing (e.g., only a question is provided without an answer or rationale), skip that entry.
        8. If the provided input is in an unknown or unexpected format and no valid triplets are found, return an empty JSON list: `[]`.

        ### Example Input (Positive Example):
        QAR NO 1  
        QUESTION: What is the person doing?,  
        ANSWER: Cycling.,  
        RATIONALE: The image shows a person riding a bicycle on a trail in a forest, indicating they are cycling.

        QAR NO 2  
        QUESTION: Why is the person cycling in the forest?,  
        ANSWER: For recreation or exercise.,  
        RATIONALE: Cycling in natural settings like forests is often associated with recreational or fitness activities.

        ### Example Output:
        [
          {"question": "What is the person doing?", "answer": "Cycling.", "rationale": "The image shows a person riding a bicycle on a trail in a forest, indicating they are cycling."},
          {"question": "Why is the person cycling in the forest?", "answer": "For recreation or exercise.", "rationale": "Cycling in natural settings like forests is often associated with recreational or fitness activities."}
        ]


        ### Example Input (Negative Example):
        I'm sorry, but I cannot generate an answer for your query. The details provided do not seem relevant to the image, and the input lacks the necessary context for a meaningful response. Kindly provide a query that directly corresponds to the image.

        ### Example Output:
        []

        DO NOT copy-paste the example inputs-outputs. They are solely for understanding the format and quality expectations.
        
        If no valid QAR triplets are found or the input format is not recognized, return an empty list: `[]`. DO NOT attempt to hallucinate or infer missing data.
        DO NOT include any additional text, explanation, or code outside the JSON list in the response.
        
        ### Handling Missing or Invalid Input:
        - If no valid QAR triplets are present in the input, return: `[]`.
        - For input text that contains explanations, acknowledgments, or irrelevant details but no actual QAR triplets, return an empty list: `[]`.
        - Do not generate or infer QAR triplets from incomplete or unrelated text."""
     
    query = """You will be provided with a text response in the format of structured QAR triplets (e.g., such as QAR NO, QUESTION, ANSWER, RATIONALE). Your task is to:
        1. Parse the text to extract valid QAR triplets into a JSON-parseable list.
        2. Each QAR triplet should be represented as an object with the following fields:
          - "question": The question text.
          - "answer": The corresponding answer.
          - "rationale": The explanation or justification for the answer.

        ### Instructions:
        - For each valid QAR triplet in the input, create a corresponding JSON object.
        - Skip entries if any part of the triplet (question, answer, or rationale) is missing.
        - The input structure may differ from the given example structure (e.g., different separators, label formatting, or order). Be flexible and adapt to process such variations while maintaining accuracy.
        - If the input format is unknown or no valid triplets are found, return an empty JSON list: `[]`.
        - Do not modify or hallucinate the content; only parse what is explicitly provided.

        ### Example Input (Positive Example):
        QAR NO 1  
        QUESTION: What is the purpose of the vehicle?,  
        ANSWER: To transport goods.,  
        RATIONALE: The vehicle depicted is a van, commonly used for transporting goods in urban areas.

        QAR NO 2  
        QUESTION: Where is the vehicle located?,  
        ANSWER: In a parking lot.,  
        RATIONALE: The image shows the vehicle parked in a designated parking area.

        ### Example Output:
        [
          {"question": "What is the purpose of the vehicle?", "answer": "To transport goods.", "rationale": "The vehicle depicted is a van, commonly used for transporting goods in urban areas."},
          {"question": "Where is the vehicle located?", "answer": "In a parking lot.", "rationale": "The image shows the vehicle parked in a designated parking area."}
        ]


        ### Example Input (Negative Example):
        I'm sorry, but I cannot generate an answer for your query. The details provided do not seem relevant to the image, and the input lacks the necessary context for a meaningful response. Kindly provide a query that directly corresponds to the image.

        ### Example Output:
        []

        ### Handling Missing or Invalid Input:
        - If no valid QAR triplets are present in the input, return: `[]`.
        - For input text that contains explanations, acknowledgments, or irrelevant details but no actual QAR triplets, return an empty list: `[]`.
        - Do not generate or infer QAR triplets from incomplete or unrelated text.
        
        DO NOT copy-paste the example inputs-outputs. They are solely for understanding the format and quality expectations.
        DO NOT include any additional text, explanation, or code outside the JSON list in the response."""

    messages = [
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": query},
        { 
            "role": "user", 
            "content": f"Here is the input text: {qar_text}.\n\nParse the input text and return a JSON list of QAR triplets. If no valid triplets are found, return: `[]`"
        },
    ]

    outputs = slm(messages)
    output = outputs[0]["generated_text"][-1]['content']
    output = clean_out_json_output(output)
    return output

def postprocess_judgement_details(slm, judgement_text):
    system_prompt = """You are a helpful assistant responsible for converting evaluation outputs from a judge model into a consistent, JSON-structured format. Your input will contain evaluations of a question based on specific criteria, with scores, justifications, failures, and evolution methods presented in text form.

        ### Task:
        1. Parse the input text and extract all relevant details.
        2. Convert the extracted details into the following JSON format:
        ```json
        {
            "scores": {
                "commonsense": {
                    "value": <integer>,
                    "justification": "<string>"
                },
                "physical_world": {
                    "value": <integer>,
                    "justification": "<string>"
                },
                "visual_understanding": {
                    "value": <integer>,
                    "justification": "<string>"
                },
                "reasoning": {
                    "value": <integer>,
                    "justification": "<string>"
                },
                "complexity": {
                    "value": <integer>,
                    "justification": "<string>"
                }
            },
            "total_justification": "<string>",
            "failures": "<string or null>",
            "evolution_method": "<string>"
        }

        ### Instructions:
        1.Ensure Correct Parsing:
          -Extract scores and justifications for each criterion (commonsense, physical_world, visual_understanding, reasoning, complexity).
          -Capture the total_justification, failures, and evolution_method as presented in the input.
        2. DO NOT infer or hallucinate missing details. Use only the explicitly provided information.
        3. The input structure may differ from the given example structure (e.g., different separators, label formatting, or order). Be flexible and adapt to process such variations while maintaining accuracy.
        4. If any field is missing in the input, return an error message in JSON format:
            ```json
            {
                "error": "Incomplete or invalid input structure"
            }
        5. Output a Valid JSON Object: Ensure the output is strictly JSON-compliant and contains no additional text, comments, or formatting issues. Each field should be correctly extracted and formatted as per the schema.

        ### Example Input (for valid input):
        ### Scores:
        - **Commonsense**: 10
          - The question assumes that the man is traveling, which is a reasonable assumption given the context of him carrying luggage.
        - **Physical World**: 5
          - The question does not require specific knowledge of the physical world beyond understanding that people travel with luggage.
        - **Visual Understanding**: 10
          - The question directly relates to the visual content of the image, where the man is seen with luggage.
        - **Reasoning**: 20
          - The question requires reasoning to infer the man's destination based on his actions and the context.
        - **Complexity**: 10
          - The question is straightforward but requires some inference.

        **Justification**:
        The question is well-rounded, requiring both commonsense knowledge and reasoning capabilities. It directly relates to the visual content of the image and does not overly complicate the task.

        **Failures**:
        None identified.

        **Evolution Method**:
        To improve the question without compromising its quality, one could add more context or details that would make the inference more challenging. For example, asking about the man's specific destination (e.g., 'What city is he heading to?') would increase the complexity while still being relevant to the image.

        ### Example Output:
        {
            "scores": {
                "commonsense": {
                    "value": 10,
                    "justification": "The question assumes that the man is traveling, which is a reasonable assumption given the context of him carrying luggage."
                },
                "physical_world": {
                    "value": 5,
                    "justification": "The question does not require specific knowledge of the physical world beyond understanding that people travel with luggage."
                },
                "visual_understanding": {
                    "value": 10,
                    "justification": "The question directly relates to the visual content of the image, where the man is seen with luggage."
                },
                "reasoning": {
                    "value": 20,
                    "justification": "The question requires reasoning to infer the man's destination based on his actions and the context."
                },
                "complexity": {
                    "value": 10,
                    "justification": "The question is straightforward but requires some inference."
                }
            },
            "total_justification": "The question is well-rounded, requiring both commonsense knowledge and reasoning capabilities. It directly relates to the visual content of the image and does not overly complicate the task.",
            "failures": null,
            "evolution_method": "To improve the question without compromising its quality, one could add more context or details that would make the inference more challenging. For example, asking about the man's specific destination (e.g., 'What city is he heading to?') would increase the complexity while still being relevant to the image."
        }

        ### Handling Invalid or Missing Input: If the input is unrelated, incomplete, or does not match the expected structure, return:
        {
            "error": "Incomplete or invalid input structure"
        }

        DO NOT copy-paste the example inputs-outputs. They are solely for understanding the format and quality expectations.
        DO NOT include any additional text, explanation, or code outside the JSON object in the response."""

    messages = [
        { "role": "system", "content": system_prompt },
        { 
            "role": "user", 
            "content": f"Here is the input text: {judgement_text}.\n\nParse the input text and return a JSON object (follow the detailed instructions and rules stated in system prompt)."
        },
    ]

    outputs = slm(messages)
    output = outputs[0]["generated_text"][-1]['content']
    cleaned_output = clean_out_json_output(output)
    valid_aspects = ["commonsense", "physical_world", "visual_understanding", "reasoning", "complexity"]
    try:
        parsed_output = json.loads(cleaned_output)
        if "scores" in parsed_output:
            total_score = sum(
                parsed_output["scores"][aspect]["value"] for aspect in valid_aspects if aspect in parsed_output["scores"]
            )
            parsed_output["total_score"] = total_score
        else:
            # Do not consider the score 
            parsed_output["total_score"] = -1
        parsed_output["evolution_method"] = "Not given" if "evolution_method" not in parsed_output else parsed_output["evolution_method"]
    except json.JSONDecodeError:
        print(f'Error: Could not parse syn_qar')
        parsed_output = {
            "total_score": -1,
            "evolution_method": "Not given"
        }

    return parsed_output

def clean_out_json_output(json_string):
    lines = json_string.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines[-1].startswith("```"):
        lines = lines[:-1]
    cleaned_json = "\n".join(lines).strip()
    
    return cleaned_json

# TODO
def generate_sample_data(image_data, file_name, slm, mllm, image_ids = []):
    sample_syn_qars = []
    sample_count = 0
    if len(image_ids) == 0:
        for i, image_details in enumerate(image_data):
            if sample_count > 50:
                file_path = '/teamspace/studios/this_studio/syn_data_gen_genetic/syn_data_gen_genetic/sample_q_no_rules_zero_shot.json'
                with open(file_path, 'w') as json_file:
                    json.dump(sample_syn_qars, json_file, indent=4)
                break
            questions = mllm.generate(image_details["image"], use_evol_prompt=False, questions=None, evolvable_questions=[], max_new_tokens=300)
            syn_qars = mllm.generate(image_details["image"], use_evol_prompt=False, questions=questions, evolvable_questions=[], max_new_tokens=1000)
            structured_syn_qars = postprocess_qars(slm, syn_qars)
            try:
                syn_qars = json.loads(structured_syn_qars)
                sample_syn_qars.append({
                    "serial": i,
                    "syn_qars": syn_qars
                })
                sample_count += 1
            except json.JSONDecodeError:
                print(f'Error: Could not parse syn_qars for {i}-th training image, moving to next image.')
                continue
    else:
        file_path = f'/teamspace/studios/this_studio/syn_data_gen_genetic/syn_data_gen_genetic/{file_name}'

        cc = 0
        for k, idx in enumerate(image_ids):
            print(f"SAMPLE {k+1}:")
            questions = mllm.generate(image_data[idx]["image"], use_evol_prompt=False, questions=None, evolvable_questions=[])
            print(f"Questions are generated...")
            syn_qars = mllm.generate(image_data[idx]["image"], use_evol_prompt=False, questions=questions, evolvable_questions=[])
            print(f"Answers and rationales are generated...")
            structured_syn_qars = postprocess_qars(slm, syn_qars)
            print(f"Non-Cleaned Output structure is generated...")
            try:
                structured_syn_qars = structured_syn_qars.strip('` \n')
                if structured_syn_qars.startswith('json'): 
                    structured_syn_qars = structured_syn_qars[4:].strip()
                syn_qars = json.loads(structured_syn_qars)
                sample_syn_qars.append({
                    "serial": idx,
                    "syn_qars": syn_qars
                })
                print(f"Output structure is cleaned...")
                #sample_count += 1
            except json.JSONDecodeError:
                print(f'Error: Could not parse syn_qars for {k+1}-th training image, moving to next image.')
                continue
            #cc += 1

        with open(file_path, 'w') as json_file:
            json.dump(sample_syn_qars, json_file, indent=4)            

    return sample_syn_qars


def load_json_file(file_name):
    file_path = f'/teamspace/studios/this_studio/syn_data_gen_genetic/syn_data_gen_genetic/{file_name}'
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def convert_and_upload_to_hf(qars, repo_name, create_dataset=True):
    if create_dataset:
        required_keys = [ "original_question_id", "question", "answer", "rationales", "image" ]
    else:
        required_keys = [ "original_question_id", "question", "answer", "rationales", "image", "synthetic_question_id", "choices", "correct_choice_idx" ]
    
    if not isinstance(qars, list) or not all(isinstance(qar, dict) for qar in qars):
        raise ValueError("Input `qars` must be a list of dictionaries.")
    if not qars:
        raise ValueError("Input `qars` cannot be an empty list.")

    validated_qars = []
    for qar in qars:
        validated_qars.append({key: qar.get(key, [] if key in {"rationales", "choices"} else "") for key in required_keys})

    
    formatted_qars = {key: [qar[key] for qar in validated_qars] for key in required_keys}
    dataset = Dataset.from_dict(formatted_qars)
    
    # Add synthetic question ID if create_dataset is True
    if create_dataset:
        """
        dataset = dataset.map(
            lambda qar: {**qar, "synthetic_question_id": secrets.token_urlsafe(19)[:25]}
        )
        """
        dataset = dataset.map(
            lambda qar: {"synthetic_question_id": secrets.token_urlsafe(19)[:25]}
        )
    
    api = HfApi()
    user_name = api.whoami()["name"]
    repo_id=f"{user_name}/{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset")
    dataset.push_to_hub(repo_id)
    print(f"Dataset is loaded to repo: {repo_id}")
    return

def upload_batch_to_hub(batch_num, batch_output, repo_name, private=True):

    if len(batch_output["question"]) == 0:
        print("Empty batch skipped.")
        return

    batch_dataset = Dataset.from_dict(batch_output)

    batch_dataset.push_to_hub(repo_name, private=private)
    print(f"Batch {batch_num+1} uploaded to {repo_name} successfully.")

"""
    for i, qar_text in enumerate(qar_texts):
        qars = postprocess_qars(slm, qar_text)
        print(f"\nSAMPLE {i+1}: ")
        try:
            syn_qars = json.loads(qars)
        except json.JSONDecodeError:
            #print(f'Error: Could not parse syn_qars for {random_i}-th training image, moving to next mllm.')
            continue
        print(type(syn_qars))
        if len(syn_qars) != 0:
            print(type(syn_qars[0]))
        print("xxxxxxxxxxxxxxxxxxxxxxxx")
    """ 