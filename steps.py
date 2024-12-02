import json

import hashlib
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import postprocess_qars, postprocess_judgement_details

# QAR generation
def generate_qars(generator_mllms, slm, image, image_details):
    new_fields = {"image": image_details["image"], "original_question_id": image_details["question_id"]}
    syn_qars_details = {}
    for i, mllm in enumerate(generator_mllms):
        questions = mllm.generate(image, use_evol_prompt=False, questions=None, evolvable_questions=[])
        syn_qars = mllm.generate(image, use_evol_prompt=False, questions=questions, evolvable_questions=[])
        syn_qars = postprocess_qars(slm, syn_qars)
        try:
            syn_qars = json.loads(syn_qars)
        except json.JSONDecodeError:
            print(f'Error: Could not parse syn_qars moving to next mllm.')
            continue

        syn_qars = [
            {**qar, 'rationales': [qar['rationale']], 'rationale': None} for qar in syn_qars
        ]

        syn_qars = [{k: v for k, v in qar.items() if k != "rationale"} for qar in syn_qars]
        
        syn_qars = [{**qar, **new_fields} for qar in syn_qars]
            
        syn_qars_details[f'mllm_{i+1}'] = {}
        for j, qar in enumerate(syn_qars):
            syn_qars_details[f'mllm_{i+1}'][f'qar_{j+1}'] = qar
    return syn_qars_details


# TODO: improve system prompt
# TODO: add json output cleaner
# Initial judgement/evaluation
def judge_qars(judge_mllm, slm, image, syn_qars_details):
    for mllm in syn_qars_details:
        judged_qars = []
        syn_qars = []
        for qar in syn_qars_details[mllm]:
            syn_qars.append(syn_qars_details[mllm][qar])

        judged_qars = judge_mllm.evaluate(syn_qars, image)

        i = 0
        for qar in syn_qars_details[mllm]:
            syn_qars_details[mllm][qar] = judged_qars[i]
            i += 1
    return syn_qars_details

        
# TODO: add json output cleaner
# Inference verification
def verify_inference(br_mllm, slm, image, syn_qars_details):
    for mllm in syn_qars_details:
        inference_verified_qars = []
        judged_syn_qars = []
        for qar in syn_qars_details[mllm]:
            judged_syn_qars.append(syn_qars_details[mllm][qar])
        
        inference_verified_qars = br_mllm.verify_inference(judged_syn_qars, image)

        i = 0
        for qar in syn_qars_details[mllm]:
            syn_qars_details[mllm][qar] = inference_verified_qars[i]
            i += 1
    return syn_qars_details


def deduplicate_qars(qars):
    unique_qars = []
    seen_qar_hashes = set()
    # step 1: remove exact copies of qars
    for qar in qars:
        key = f'{qar["question"]}, {qar["answer"]}, {qar["rationales"]}'
        qar_hash = hashlib.md5(key.encode()).hexdigest()
        if qar_hash not in seen_qar_hashes:
            unique_qars.append(qar)
            seen_qar_hashes.add(qar_hash)

    # step 2: remove highly similar qars (1 of them)
    nonsimilar_qars = []
    seen_qars = []
    for qar in unique_qars:
        qar_1_text = f'{qar["question"]}, {qar["answer"]}, {qar["rationales"]}'
        qar_is_duplicate = False
        for seen_qar in seen_qars:
            qar_2_text = f'{seen_qar["question"]}, {seen_qar["answer"]}, {seen_qar["rationales"]}'
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


# Evolution method generation
def generate_evol_method(final_judge, image, qars):
    qars_with_evol = final_judge.evaluate(image, qars)
    return qars_with_evol

#TODO: Evolve QARs
def evolve_qars():
    pass