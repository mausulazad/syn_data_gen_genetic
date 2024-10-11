import requests
import torch
from PIL import Image

from utils import load_and_preprocess_dataset, setup_models

def generate_qars(dataset, generator_models, judge_model, br_model):
    # step 0: load and preprocess dataset
    #data = load_and_preprocess_dataset(dataset)
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # step 1: load models
    generator_mllms = setup_models(generator_models, judge_model, br_model)

    # step 2: generate qars
    # query = "Can you describe the activity of the animal in context of the image?"
    query = "Can you generate 3 non-trivial, diverse questions and corresponding answers based on the image without hallucinating? Keep the answers precise and short (no over explanation).Return the question-answer pairs in a list following this structure: [{'question': <question>, 'answer': <answer>}]. Return only the list of JSON objects, nothing else."
    generator_mllms[0].generate(image, query)

    # step 3: judge them using judge mllm

    # step 4: do inference verification using backward reasoner mllm

    # step 5: keep all filtered in quality qars

    # step 6: return qars stored in step 5