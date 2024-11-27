import json

from utils import postprocess_qars, postprocess_judgement_details

# QAR generation
def generate_qars(generator_mllms, slm, image):
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


# Evolution method generation
def generate_evol_method(final_judge, image, qars):
    qars_with_evol = final_judge.evaluate(image, qars)
    return qars_with_evol

#TODO: Evolve QARs
def evolve_qars():
    pass
"""