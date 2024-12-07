import requests
import json
import random
import copy
import pprint
import torch
from PIL import Image

import sys
import warnings
import os

from utils import load_and_preprocess_dataset, setup_slm, setup_models, postprocess_qars, setup_final_judge, get_gpu_details, load_json_file, generate_sample_data, postprocess_judgement_details, convert_and_upload_to_hf

from steps import generate_qars, evolve_qars, judge_qars, verify_inference, deduplicate_qars, generate_evol_method


def build_synthetic_dataset(dataset, generator_models, judge_model, br_model):
    # Load MLLMs
    # Setup a SLM (Llama-3.2 1B/3B) for output structure related post-processing
    slm = setup_slm()
    # generator_mllms, judge_mllm, br_mllm = setup_models(generator_models, judge_model, br_model)
    final_judge = setup_final_judge(model="llava_critic")

    # Load aokvqa/scienceqa dataset
    seed_dataset = load_and_preprocess_dataset(dataset)
    #random_i = random.randint(0, len(image_data)-1)
    #image = image_data[random_i]["image"]
    #image = image_data[15572]["image"]
    #image = image_data[9344]["image"]

    #TODO: append object with options
    #slm = None
    synthetic_qars = []
    # while runs < 5:
    for i, data in enumerate(seed_dataset):
        tries = 0
        #evolvable_questions = []
        evolvable_questions = [
            ('What time of day is it in the image?', 'Add more context or details to increase the complexity without compromising other aspects.'), 
            ('How many suitcases does the man have?', 'No significant evolution is needed for this question as it is already clear and concise. However, if the goal is to increase complexity, one could ask about the total weight of the suitcases or the purpose of the man carrying them, which would require more reasoning and context..'), 
            ("What is the man's destination?", 'To improve the question, add elements that require specific visual details to answer, such as asking about the type of luggage or the specific setting, to enhance complexity without compromising other aspects..')
        ]
        
        #while tries <= 4:
        if len(evolvable_questions) == 0:
            syn_qars_details = generate_qars(generator_mllms, slm, data["image"], data)
        else:
            syn_qars_details = evolve_qars(generator_mllms, slm, data["image"], data, evolvable_questions)
        
        #syn_qars_details = judge_qars(judge_mllm, slm, data["image"], syn_qars_details)
        #syn_qars_details = verify_inference(br_mllm, slm, data["image"], syn_qars_details)

        # Filter out qars that do not pass inference verifier
        syn_qar_bucket = []
        for mllm in syn_qars_details:
            for qar in syn_qars_details[mllm]:
                #if (syn_qars_details[mllm][qar]["br_score"] > 0.7):
                syn_qar_bucket.append(syn_qars_details[mllm][qar])

        # De-duplicate initially filtered qars
        unique_qars = deduplicate_qars(syn_qar_bucket)

        synthetic_qars.extend(unique_qars)

        syn_qars_with_evol = generate_evol_method(final_judge, data["image"], unique_qars)
            
            
        # TODO: Move to 'steps' file
        evolvable_questions = []
        for syn_qar in syn_qars_with_evol:
            if syn_qar["judgement_details"]["total_score"] >= 80:
                # TODO choices = generate_options(slm, question, answer)
                synthetic_qars.append({
                    "id": data["id"],
                    "image": data["image"],
                    "question": syn_qar["question"],
                    "direct_answer": syn_qar["answer"],
                    "choices": choices
                })
            else:
                evolvable_questions.append((syn_qar["question"], syn_qar["judgement_details"]["evolution_method"]))

        if len(evolvable_questions) == 0:
            break
        else:
            tries += 1

        #if i % 20 == 0:
        #    print(f"QAR for {i} images are generated")
        
        #if i >= 700:
        #    break
        break

    # Store in huggingface repo
    #repo_name = "syn_dataset_no_evolution_single_run"
    #convert_and_upload_to_hf(synthetic_qars, repo_name)


# TODO: Later, make parallel inference calls (as possible)
def generate_qars(dataset, generator_models, judge_model, br_model):
    image_data = load_and_preprocess_dataset(dataset)
    final_judge = setup_final_judge(model="llava_next")
    slm = setup_slm()
    # Try to generate evolution details for LLaVA-Critic
    # Then, if time allows do the same using LLaVA-Next
    sample_syn_qars = load_json_file(file_name='sample_q_few_shot_cot_llama_run_4.json') 
    # Create a separate judgement block and place it there
    kk = 0
    judged_syn_qars = []
    for i, syn_qar in enumerate(sample_syn_qars):
        print(f"JUDGEMENT {i+1}:")
        idx, qars = syn_qar["serial"], syn_qar["syn_qars"]
        qars = final_judge.evaluate(image_data[idx]["image"], qars)

        for j, qar in enumerate(qars):
            qars[j]["judgement_details"] = postprocess_judgement_details(slm, qar["judgement_details"])

        judged_syn_qars.append({
            "serial": idx,
            "syn_qars": qars
        })

    with open("judge_llava_next_generator_llama_v2.json", "w") as file:
        json.dump(judged_syn_qars, file, indent=4)

    
    
    """
    image_ids = []
    for syn_qar in sample_syn_qars:
        image_ids.append(syn_qar["serial"])
        print(syn_qar)
        break
    
    sample_syn_qars = generate_sample_data(image_data, 'sample_q_few_shot_cot_molmo.json', slm, generator_mllms[0], image_ids)      
    print(len(sample_syn_qars))

    #final_judge_mllm.evaluate(unique_qars, image)

    #print(random_i)
    # Remove after testing  
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)

    for idx in image_ids:
        image = image_data[idx]["image"]
    
    
    
    # TODO: Place this entire commneted block inside the for loop
    evolvable_questions = []
    tries = 0
    # LATER: Make 3 tries
    while tries < 1:
        # TODO: Check how mllms (existing ones perform in evolving task) 
        # Step 2: Generate qars
        #query = "Can you describe the activity of the animal in context of the image?"
        #query = "Can you generate 3 non-trivial, diverse questions and corresponding answers based on the image without hallucinating? Keep the answers precise and short (no over explanation).Return the question-answer pairs in a list following this structure: [{'question': <question>, 'answer': <answer>}]. Return only the list of JSON objects, nothing else."
        
        use_evol_prompt = False
        if len(evolvable_questions) != 0:
            use_evol_prompt = True
        
        # HEREEE
        evolvable_questions = []
        unique_qars = final_judge_mllm.evaluate(unique_qars, image)
        for unique_qar in unique_qars:
            # Current qar can be stored in final dataset
            if unique_qar["evaluation"]["score"] >= 80:
                # TODO: Generate options (use phi3-mini like Moshiur did)
                synthetic_qars.append({
                    "question": unique_qar["question"],
                    "answer": unique_qar["answer"],
                    "rationale": unique_qar["rationale"]
                })

            # This additional check is done, since the score is not always reliable (arithmetic mistakes may occur by MLLM)
            # Check if current qar can be further evolved
            failures = unique_qars["evaluation"]["failures"]
            evol_text = unique_qar["evaluation"]["evolution_method"]
            if  evol_text and (('None' not in failures) or (evol_text != 'None') or (evol_text is not None)):
                evolvable_questions.append({
                    "question": unique_qar["question"],
                    "evolution_inst": unique_qar["evaluation"]["evolution_method"]
                })

        # All qars are evolved as much as possible
        if len(evolvable_questions) == 0:
            break

        #pprint.pprint(evolvable_questions)
        
        tries += 1
    
    #TODO: Apply dedup again on final synthetic dataset

    #TODO: Store in huggingface
    """


"""
sample_syn_qars = load_json_file(file_name='judge_llava_critic_generator_llava_next.json') 

    for i, image_details in enumerate(sample_syn_qars):
        evolvable_questions = []
        for item in image_details['syn_qars']:
            question = item.get('question')
            evolution_method = item.get('judgement_details', {}).get('evolution_method')
        
            if question and evolution_method:
                evolvable_questions.append((question, evolution_method))
        
        break
    
    sample_syn_qars = load_json_file(file_name='sample_q_few_shot_cot_llama_run_4.json') 
    
    image_ids = []
    for syn_qar in sample_syn_qars:
        image_ids.append(syn_qar["serial"])

    sample_syn_qars = load_json_file(file_name='judge_llava_critic_generator_llava_next.json') 

    evolvable_questions = {}
    for i, syn_qar in enumerate(sample_syn_qars):
        #print(syn_qar["syn_qars"][0]["judgement_details"]["evolution_method"])
        evol_pairs = [(qar["question"], qar["judgement_details"]["evolution_method"]) for qar in syn_qar["syn_qars"] if "evolution_method" in qar["judgement_details"]]
        evolvable_questions[syn_qar["serial"]] = evol_pairs

    syn_data_qars_details = []
    for serial in evolvable_questions:
        syn_data_qars = {
            "serial": serial
        } 
        syn_data_qars["syn_qars"] = evolve_qars(generator_mllms, slm, seed_dataset[serial]["image"], seed_dataset[serial], evolvable_questions[serial])
        syn_data_qars_details.append(syn_data_qars)

    with open("evolved_qars_llava_next_run_4.json", "w") as file:
        json.dump(syn_data_qars_details, file, indent=4)
    """