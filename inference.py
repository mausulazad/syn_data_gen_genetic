import requests
import json
import time
import random
import copy
import pprint
import torch
from PIL import Image

import sys
import warnings
import os

import numpy as np

#from multiprocess import set_start_method
#import torch.multiprocessing as mp
from multiprocess import set_start_method
from torch.utils.data import DataLoader

from utils import (
    load_and_preprocess_dataset,
    setup_slm,
    #setup_models,
    setup_generator_models,
    postprocess_qars,
    setup_final_judge,
    setup_jury_poll,
    get_gpu_details,
    load_json_file,
    generate_sample_data,
    postprocess_judgement_details,
    convert_and_upload_to_hf,
    setup_synthesizer_llm,
    synthesize_evol_methods,
    upload_batch_to_hub
)

#from option_gen import generate_options

from steps import generate_qars, evolve_qars, eval_qars, verify_inference, deduplicate_qars

model_card = [
    # slm
    {
        "name": "slm",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "device": "cpu"
    },
    # generator mllms
    {
        "name": "llama_32",
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "device": "cpu"
    },
    {
        "name": "llava_next",
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "device": "cpu"
    },
    {
        "name": "molmo",
        "model_id": "allenai/Molmo-7B-D-0924",
        "device": "cpu"
    },
    # jury mllms
    {
        "name": "llava_critic",
        "model_id": "lmms-lab/llava-critic-7b",
        "device": "cpu"
    },
    {
        "name": "prometheus_vision",
        "model_id": "kaist-ai/prometheus-vision-13b-v1.0",
        "device": "cpu"
    },
    {
        "name": "qwen2_vl",
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "device": "cpu"
    },
    # synthesizer llm
    {
        "name": "qwen_25",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cpu"
    }
]

slm, generator_mllms, jury_mllms, synthesizer = None, [], [], None

def allocate_gpu(model_card):
    devices = torch.cuda.device_count()
    print(devices)
    if devices > 0:
        for i, _ in enumerate(model_card):
            model_card[i]["device"] = f"cuda:{(i) % devices}"
    return model_card

def config_models(model_card, generator_model_names, jury_model_names):
    # Distribute available gpus to models
    model_card = allocate_gpu(model_card)
        
    # Load models (need to do this for each distinct process, since processes can not access parent process variables)
    slm = setup_slm("slm", model_card)
    generator_mllms = setup_generator_models(generator_model_names, model_card)
    jury_mllms = setup_jury_poll(jury_model_names, model_card)
    synthesizer = setup_synthesizer_llm("qwen_25", model_card)
    
    return {
        "parser": slm,
        "generator_mllms": generator_mllms,
        "jury_mllms": jury_mllms,
        "synthesizer": synthesizer
    }

def build_dataset(dataset, generator_model_names, jury_model_names):
    # Load aokvqa/scienceqa dataset
    seed_dataset = load_and_preprocess_dataset(dataset)
    
    # USE SAMPLES FOR TESTING & DEBUGGING. LATER RUN THIS ON ENTIRE DATASET
    # num_samples = len(seed_dataset)
    #print(num_samples)
    # sample_indices = np.random.choice(num_samples, size=4, replace=False)
    # seed_dataset = seed_dataset.select(sample_indices)
    
    gen_model_count = sum(1 for model in model_card if model.get("name") in generator_model_names)
    jury_model_count = sum(1 for model in model_card if model.get("name") in jury_model_names)
    slm_model_exists = any(model.get("name") == "slm" for model in model_card)
    synthesizer_model_exists = any(model.get("name") == "qwen_25" for model in model_card)
    
    if not ((gen_model_count == len(generator_model_names)) and (jury_model_count == len(jury_model_names)) and slm_model_exists and synthesizer_model_exists):
        print("1 or more given models' config details is not available.")
        return

    model_details = config_models(model_card, generator_model_names, jury_model_names)
    
    batch_size = 5
    for batch_num, batch_output in enumerate(
        seed_dataset.map(
            build_synthetic_dataset,
            batched=True,
            batch_size=batch_size,
            with_rank=True,
            # num_proc=1,
            num_proc=torch.cuda.device_count(),
            fn_kwargs={
                "model_card": model_card,
                "model_details": model_details,
                "generator_model_names": generator_model_names,
                "jury_model_names": jury_model_names
            }
        )
    ):
        upload_batch_to_hub(
            batch_num,
            batch_output=batch_output,
            # TODO: replace with a original repo name
            repo_name="mausul/test_demo",
            private=True
        )
        print(batch_output)


def build_synthetic_dataset(batch, rank, model_card, model_details, generator_model_names, jury_model_names):
    """
    try:
        mp.set_start_method("spawn", force=True)  # Ensure correct start method
    except RuntimeError:
        pass  # Ignore if already set
    """
    
    #model_details = config_models(model_card, generator_model_names, jury_model_names)
    #manager = mp.Manager()
    
    parser = model_details["parser"]
    generator_mllms = model_details["generator_mllms"]
    jury_mllms = model_details["jury_mllms"]
    synthesizer = model_details["synthesizer"]
    
    quality_qars = []
    evolvable_questions = []
    max_tries = 2
    current_try = 1
    while current_try <= max_tries:
        # shared across all processes
        #qars = manager.list()
        qars = []
        
        # DO NOT USE: multiprocessing based code
        """
        processes = []
        for process_id, model in enumerate(generator_mllms):
            if len(evolvable_questions) == 0:
                p = mp.Process(target=generate_qars, args=(process_id, batch, model, qars, parser))
                p.start()
                processes.append(p)
            else:
                p = mp.Process(target=evolve_qars, args=(process_id, evolvable_questions, model, qars, parser))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
        """
        
        # USE THIS CODE SNIPPET
        for idx, model in enumerate(generator_mllms):
            
            # we'll generated the qar for the very first time
            if len(evolvable_questions) == 0:
                qars.extend(generate_qars(batch, model, parser))
            else:
                qars.extend(evolve_qars(evolvable_questions, model, parser))

        all_qars = []
        
        # we have multiple qars for one image.
        # we'll map one-one mapping image-qar from one-to-many (image-qars)
        for image, syn_qars in qars:
            #print(f"[RESULT] Process {process_id}: {model_name} -> '{input_text}' => '{output_text}'")
            for qar in syn_qars:
                all_qars.append((image, qar))

        # TODO push code (comment out, not remove)

        # jury time to evelauate the first generated qars
        evolvable_questions = []
        for idx, qar_details in enumerate(all_qars):
            image, qar = qar_details
            #evol_methods = manager.list()
            evol_methods = []
            jury_processes = []
            
            
            # DO NOT USE: multiprocessing based code
            """
            for process_id, model in enumerate(jury_mllms):
                p = mp.Process(target=eval_qars, args=(process_id, qar_details, model, evol_methods, parser))
                p.start()
                jury_processes.append(p)

            for p in jury_processes:
                p.join()
            """

            # enumerate jury models and pass qars to each model & get the evol method
            # so if no. of jury is three (3) we'd have three (3) evol methods
            for model in jury_mllms:
                evol_methods.extend(eval_qars(qar_details, model, parser))
                            

            # we will asses if we get the proper evol method 
            # if yes we'd store the score and method 
            evolution_methods = []
            scores = []
            for evol_method_details in evol_methods:
                # print('---------')
                # print(evol_method_details)
                # print('-----------------')
                
                # a condition to discard the error evolv method, exp: prometheus vision 
                if evol_method_details["evolution_method"] != "Not given" and evol_method_details["total_score"] != -1:
                    evolution_methods.append(evol_method_details["evolution_method"])
                    scores.append(evol_method_details["total_score"])
        
            # ===========================================================================================================================
            # average the jury poll score and synthesize all evol methods 
            jury_score = 0
            if len(scores) > 0:
                jury_score = sum(scores)/len(scores)

            # synthesize evol_methods
            synthesized_evol_method = synthesize_evol_methods(synthesizer, qar.get("question", "None generated"), evolution_methods)
            # ===========================================================================================================================


            # checking if the target score is achieved they are judged as quality qar
            if jury_score > 90:
                quality_qars.append({
                    "question": qar.get("question", "None generated"),
                    "answer": qar.get("answer", "None generated"),
                    "rationales": qar.get("rationales", "None generated"),
                    "image": image,
                    "jury_score": jury_score,
                    "evol_method": synthesized_evol_method
                })

            
            # non quality qars are going to be evolved
            else:
                evolvable_questions.append({
                    "question": qar.get("question", "None generated"),
                    "answer": qar.get("answer", "None generated"),
                    "rationales": qar.get("rationales", "None generated"),
                    "image": image,
                    "jury_score": jury_score,
                    "evol_method": synthesized_evol_method
                
                })
            
        if len(evolvable_questions) == 0:
            break
 
        current_try += 1

    if len(quality_qars) != 0: 
        qar_map = {key: [item[key] for item in quality_qars] 
                          for key in quality_qars[0]}
    else:
        qar_map = {
            "question": [],
            "answer": [],
            "rationales": [],
            "image": [],
            "jury_score": [],
            "evol_method": []
        }
    
    return qar_map
    #return quality_qars
    

"""
#def build_synthetic_dataset(dataset, generator_models, judge_model, br_model):
    # Load MLLMs
    # Setup a SLM (Llama-3.2 1B/3B) for output structure related post-processing
    slm = setup_slm()
    generator_mllms, judge_mllm, br_mllm = setup_models(generator_models, judge_model, br_model, model_card)
    final_judge = setup_final_judge(model="llava_critic")
    #juries = setup_jury_poll(["prometheus_vision"])
    #juries = setup_jury_poll(["llava_critic", "qwen2_vl"])
    juries = setup_jury_poll(["llava_critic", "qwen2_vl", "prometheus_vision"], model_card)
    synthesizer = setup_synthesizer_llm("qwen_25", model_card)

    # Load aokvqa/scienceqa dataset
    seed_dataset = load_and_preprocess_dataset(dataset)
    #random_i = random.randint(0, len(image_data)-1)
    #image = seed_dataset[564]["image"]
    
    synthetic_qars = []
    #slm = None
    #generator_mllms = ["molmo", "llama_32", "llava_32"]
    total_inference_time = 0
    for i, data in enumerate(seed_dataset):
        evolvable_questions = []
        evolvable_questions = [
            ('What time of day is it in the image?', 'Add more context or details to increase the complexity without compromising other aspects.'), 
            ('How many suitcases does the man have?', 'No significant evolution is needed for this question as it is already clear and concise. However, if the goal is to increase complexity, one could ask about the total weight of the suitcases or the purpose of the man carrying them, which would require more reasoning and context..'), 
            ("What is the man's destination?", 'To improve the question, add elements that require specific visual details to answer, such as asking about the type of luggage or the specific setting, to enhance complexity without compromising other aspects..')
        ]
        
        start = time.time()
        syn_qar_bucket = []
        max_runs = 3
        runs = 0
        while runs < max_runs:
            if len(evolvable_questions) == 0:
                syn_qars_details = generate_qars(generator_mllms, slm, data["image"], data)
            else:
                syn_qars_details = evolve_qars(generator_mllms, slm, data["image"], data, evolvable_questions)
        
            #syn_qars_details = judge_qars(judge_mllm, slm, data["image"], syn_qars_details)
            #syn_qars_details = verify_inference(br_mllm, slm, data["image"], syn_qars_details)

            # Filter out qars that do not pass inference verifier
            for mllm in syn_qars_details:
                for qar in syn_qars_details[mllm]:
                    #if (syn_qars_details[mllm][qar]["br_score"] > 0.7):
                    syn_qar_bucket.append(syn_qars_details[mllm][qar])

            # De-duplicate initially filtered qars
            unique_qars = deduplicate_qars(syn_qar_bucket)
        
            end = time.time()
        
            elapsed_time = end - start
            #print(f"Inference time for {i+1} image(s): {elapsed_time:.2f} seconds")
            total_inference_time += elapsed_time

            #synthetic_qars.extend(unique_qars)
        
            syn_qars_with_evol = get_jury_verdicts(juries, slm, synthesizer, data["image"], unique_qars)
            #syn_qars_with_evol = generate_evol_method(final_judge, slm, data["image"], unique_qars)

            for i, qar in enumerate(syn_qars_with_evol):
                syn_qars_with_evol[i]["judgement_details"] = postprocess_judgement_details(slm, qar["judgement_details"])
            
            
            runs += 1

            # TODO: Move to 'steps' file
            evolvable_questions = []
            for syn_qar in syn_qars_with_evol:
                if (syn_qar["avg_score"] >= 90) or ((runs == max_runs) and (syn_qar["avg_score"] >= 75)):
                    image, question, direct_answer = data["image"], syn_qar["question"], syn_qar["answer"]
                    # TODO: Check option gen and qar object update's effect on hf upload util function, bugs may exist
                    choices, correct_choice_idx = generate_options(image, question, direct_answer)
                    synthetic_qars.append({
                        "original_question_id": data["id"],
                        "image": image,
                        "question": question,
                        "direct_answer": direct_answer,
                        "choices": choices,
                        "correct_choice_idx": correct_choice_idx
                    })
                else:
                    if syn_qar["evol_method"] != "Not given":
                        evolvable_questions.append((syn_qar["question"], syn_qar["evol_method"]))
        
            if len(evolvable_questions) == 0:
                break

        if i % 20 == 19:
            print(f"qars for {i+1} images are generated...")
            print(f"No. of qars generated (till now): {len(synthetic_qars)}")
            print(f"Total inference time (till now): {total_inference_time/60:.2f} min(s)")
            print("="*80)
        
        if i % 2 == 1:
            print(f"qars for {i+1} images are generated...")
            print(f"No. of qars generated (till now): {len(synthetic_qars)}")
            print(f"Total inference time (till now): {total_inference_time/60:.2f} min(s)")
            print("="*80)
        
        if i >= 400:
            break
        #print(len(synthetic_qars))
        #break
        

    # Store in huggingface repo
    # TODO: Change repo name
    repo_name = "syn_dataset_no_evolution_multi_run_smol_v1"
    convert_and_upload_to_hf(synthetic_qars, repo_name)
"""

"""
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