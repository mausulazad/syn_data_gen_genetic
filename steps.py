import json

import hashlib
#from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from accelerate import Accelerator
accelerator = Accelerator()

from utils import postprocess_qars, postprocess_judgement_details, synthesize_evol_methods

def generate_qars(batch, model, parser):
    qars = []
    for idx, image_details in enumerate(batch['image']):

        image = image_details
        #print(f"[DEBUG] Process {process_id}: Model {model_name} processing text {idx} on {device}: '{text}'")
        questions = model.generate(image, use_evol_prompt=False, questions=None, evolvable_questions=[])
        syn_qars = model.generate(image, use_evol_prompt=False, questions=questions, evolvable_questions=[])
        syn_qars = postprocess_qars(parser, syn_qars)
        
        print('-'*50)
        print(f'Sample: {syn_qars}')
        print(f'Type of qar: {type(syn_qars)}')
        print('-'*50)

        try:
            syn_qars = json.loads(syn_qars)
        except json.JSONDecodeError:
            #accelerator.print(f"Error: Could not parse syn_qars for model {i+1}. Skipping...")
            #print(f'Error: Could not parse syn_qars moving to next mllm.')
            continue

        print('***********************After try catch************************')
        print(f'Sample: {syn_qars}')
        print(f'Type of qar: {type(syn_qars)}')
        print('-'*50)


        syn_qars = [
            {**qar, 'rationales': [qar.get('rationale', 'Not Generated')], 'rationale': None} for qar in syn_qars
        ]

        syn_qars = [{k: v for k, v in qar.items() if k != "rationale"} for qar in syn_qars]
        syn_qars = deduplicate_qars(syn_qars)
        qars.append((image, syn_qars))
    return qars

"""
# QAR generation
def generate_qars(generator_mllms, slm, image, image_details):
    generator_mllms = [accelerator.prepare(mllm) for mllm in generator_mllms]
    new_fields = {"image": image_details["image"], "original_question_id": image_details["question_id"]}
    syn_qars_details = {}
    for i, mllm in enumerate(generator_mllms):
        questions = mllm.generate(image, use_evol_prompt=False, questions=None, evolvable_questions=[])
        syn_qars = mllm.generate(image, use_evol_prompt=False, questions=questions, evolvable_questions=[])
        syn_qars = postprocess_qars(slm, syn_qars)
        try:
            syn_qars = json.loads(syn_qars)
        except json.JSONDecodeError:
            accelerator.print(f"Error: Could not parse syn_qars for model {i+1}. Skipping...")
            #print(f'Error: Could not parse syn_qars moving to next mllm.')
            continue

        syn_qars = [
            {**qar, 'rationales': [qar.get('rationale', 'Not Generated')], 'rationale': None} for qar in syn_qars
        ]

        syn_qars = [{k: v for k, v in qar.items() if k != "rationale"} for qar in syn_qars]
        
        syn_qars = [{**qar, **new_fields} for qar in syn_qars]
            
        syn_qars_details[f'mllm_{i+1}'] = {}
        for j, qar in enumerate(syn_qars):
            syn_qars_details[f'mllm_{i+1}'][f'qar_{j+1}'] = qar
    return syn_qars_details
"""


"""
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
"""

        
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
        # To avoid parsing error
        question = qar.get("question", "")
        answer = qar.get("answer", "")
        rationales = qar.get("rationales", "")
        
        key = f'{question}, {answer}, {rationales}'
        qar_hash = hashlib.md5(key.encode()).hexdigest()
        if qar_hash not in seen_qar_hashes:
            unique_qars.append(qar)
            seen_qar_hashes.add(qar_hash)

    # step 2: remove highly similar qars (1 of them)
    nonsimilar_qars = []
    seen_qars = []
    for qar in unique_qars:
        qar_1_text = f'{qar.get("question", "")}, {qar.get("answer", "")}, {qar.get("rationales", "")}'
        qar_is_duplicate = False
        for seen_qar in seen_qars:
            qar_2_text = f'{seen_qar.get("question", "")}, {seen_qar.get("answer", "")}, {seen_qar.get("rationales", "")}'
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

"""
# TODO: DO NOT call from inference.py
# Evolution method generation
def eval_qars(final_judge, slm, image, qars):
    judgements = []
    qars_with_evol = final_judge.evaluate(image, qars)
    for i, qar in enumerate(qars_with_evol):
        judgement_details = postprocess_judgement_details(slm, qar["judgement_details"])
        judgements.append([judgement_details["total_score"], judgement_details["evolution_method"]])
    scores, evol_methods = zip(*(judgements))
    return (scores, evol_methods)
"""

"""
def get_jury_verdicts(juries, slm, synthesizer, image, qars):
    all_scores, all_evol_methods = [], []
    for i, jury in enumerate(juries):
        #print(f"JURY {i+1}:")
        judgements = eval_qars(jury, slm, image, qars)
        all_scores.append(judgements[0])
        all_evol_methods.append(judgements[1])
    qars_scores = zip(*all_scores)
    qars_evol_methods = zip(*all_evol_methods)
    for i, qar_scores in enumerate(qars_scores):
        qar_scores = [ 4*qar_score for qar_score in qar_scores if qar_score != -1]
        avg_score = sum(qar_scores) / len(qar_scores)
        qars[i]["avg_score"] = avg_score
    
    for i, qar_evol_methods in enumerate(qars_evol_methods):
        # synthesize evol methods
        qar_evol_methods = [qar_evol_method for qar_evol_method in qar_evol_methods if qar_evol_method != "Not given"]
        if len(qar_evol_methods) > 0:
            synthesized_evol_method = synthesize_evol_methods(synthesizer, qars[i]["question"], qar_evol_methods)
            qars[i]["evol_method"] = synthesized_evol_method
        else:
            qars[i]["evol_method"] = None
  
    return qars
"""

def eval_qars(qar_details, model, parser):
    evol_methods = []
    image, qar = qar_details
    evol_details = model.evaluate([qar], image)
    evol_details = evol_details[0]["judgement_details"]
    evol_details = postprocess_judgement_details(parser, evol_details)
    evol_methods.append(evol_details)
    """
    judgements.append([judgement_details["total_score"], judgement_details["evolution_method"]])
    scores, evol_methods = zip(*(judgements))
    """
    #return (scores, evol_methods)
    return evol_methods

"""
def activate_jury_poll(juries, bailiff, slm, image, qars):
    verdicts = get_jury_verdicts(juries, slm, image, qars)
    for i, qar in enumerate(qars):
        evol_methods = [verdict["evol_method"] for verdict in verdicts]
        permissible_evol_method = synthesize_evol_methods(bailiff, evol_methods)
        qars[i]["evol_method"] = permissible_evol_method
    return qars
"""


#Evolve QARs
def evolve_qars(evolvable_questions, model, parser):
    qars = []
    for idx, image_details in enumerate(evolvable_questions):
        image = image_details["image"]
        #new_fields = {"image": image_details["image"], "original_question_id": image_details["question_id"]}
        evolvable_question_details = [image_details["question"], image_details["evol_method"]]
        questions = model.generate(image, use_evol_prompt=True, questions=None, evolvable_questions=[evolvable_question_details])
        syn_qars = model.generate(image, use_evol_prompt=False, questions=questions, evolvable_questions=[])
        syn_qars = postprocess_qars(parser, syn_qars)
        try:
            syn_qars = json.loads(syn_qars)
        except json.JSONDecodeError:
            print(f'Error: Could not parse syn_qars moving to next mllm.')
            continue

        syn_qars = [
            {**qar, 'rationales': [qar.get('rationale', 'Not Generated')], 'rationale': None} for qar in syn_qars
        ]

        syn_qars = [{k: v for k, v in qar.items() if k != "rationale"} for qar in syn_qars]
        syn_qars = deduplicate_qars(syn_qars)
        qars.append((image, syn_qars))
    return qars

"""
#Evolve QARs
def evolve_qars(generator_mllms, slm, image, image_details, evolvable_questions):
    #new_fields = {"image": image_details["image"], "original_question_id": image_details["question_id"]}
    syn_qars_details = {}
    for i, mllm in enumerate(generator_mllms):
        questions = mllm.generate(image, use_evol_prompt=True, questions=None, evolvable_questions=evolvable_questions)
        syn_qars = mllm.generate(image, use_evol_prompt=False, questions=questions, evolvable_questions=[])
        syn_qars = postprocess_qars(slm, syn_qars)
        try:
            syn_qars = json.loads(syn_qars)
        except json.JSONDecodeError:
            print(f'Error: Could not parse syn_qars moving to next mllm.')
            continue

        syn_qars = [
            {**qar, 'rationales': [qar.get('rationale', 'Not Generated')], 'rationale': None} for qar in syn_qars
        ]

        syn_qars = [{k: v for k, v in qar.items() if k != "rationale"} for qar in syn_qars]
        
        #syn_qars = [{**qar, **new_fields} for qar in syn_qars]

        syn_qars_details[f'mllm_{i+1}'] = {}
        for j, qar in enumerate(syn_qars):
            syn_qars_details[f'mllm_{i+1}'][f'qar_{j+1}'] = qar
    return syn_qars_details
"""