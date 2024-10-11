import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

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

def setup_generator_models(generator_models):
    generator_mllms = []
    for model_name in generator_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            generator_mllms.append(MLLM(model, processor, inference_type='generate'))
        # TODO: add other generator mllms (i.e. llava vision, molmo)
    return generator_mllms

def setup_judge_models(judge_models):
    judge_mllms = []
    for model_name in judge_models:
        if model_name == "llama32":
            model, processor = setup_llama32()
            judge_mllms.append(Judge(model, processor, inference_type='judge'))
        # TODO: Pr 3 --> add other judge mllms (i.e. llava-critic, blip-3, sfr judge, prometheus)
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
    generator_mllms = setup_generator_models(generator_models)
    #judge_mllm = setup_judge_models([judge_model])[0]
    #br_mllm = setup_backward_reasoning_models([br_model])[0]
    
    #return (generator_mllms, judge_mllm, br_mllm)
    return (generator_mllms)



# TODO
def infer_using_llama32():
    pass

# LATER
def infer_using_llava_critic():
    pass

# LATER
def infer_using_molmo():
    pass

# LATER
def infer_using_llava():
    pass