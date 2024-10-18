import requests
import json
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
    #query = "Can you generate 3 non-trivial, diverse questions and corresponding answers based on the image without hallucinating? Keep the answers precise and short (no over explanation).Return the question-answer pairs in a list following this structure: [{'question': <question>, 'answer': <answer>}]. Return only the list of JSON objects, nothing else."
    query_1 = "Can you generate 3 non-trivial, diverse questions based on the image without hallucinating?  Each question can not have sub-questions. Return the questions in a list (comma separated) like this: [<question 1>, <question 2>, <question 3>]. Return only the list of questions, nothing else."
    query_2 = 'You will be given an image and 3 questions that are based on the image. For each question generate correct answer and corresponding brief rationale (rationale should justify briefly why the answer is correct) without hallucinating. Keep the answers precise and short (no over explanation). Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. Return only the list of JSON objects, nothing else.'
    all_syn_qars = {}
    for i, mllm in enumerate(generator_mllms):
        questions = mllm.generate(image, query_1, questions=None)
        syn_qars = mllm.generate(image, query_2, questions)
        syn_qars = json.loads(syn_qars)
        all_syn_qars[f'mllm_{i+1}'] = {}
        for j, qar in enumerate(syn_qars):
            all_syn_qars[f'mllm_{i+1}'][f'qar_{j+1}'] = qar
    
    # step 3: judge them using judge mllm

    # step 4: do inference verification using backward reasoner mllm

    # step 5: keep all filtered in quality qars

    # step 6: return qars stored in step 5