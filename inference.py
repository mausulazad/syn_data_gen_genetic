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

from utils import load_and_preprocess_dataset, setup_slm, setup_models, postprocess_qars, deduplicate_qars, setup_final_judge, get_gpu_details, load_json_file, generate_sample_data

def judge_qars(qars, image, judge_mllm):
    qars = judge_mllm.evaluate(qars, image)
    return qars

def verify_inference(qars, image, br_mllm):
    qars = br_mllm.verify_inference(qars, image)
    return qars

# TODO: Later, make parallel inference calls (as possible)
def generate_qars(dataset, generator_models, judge_model, br_model):
    synthetic_qars = []
    
    # Load aokvqa/scienceqa dataset
    # Step 0: Load and preprocess dataset
    image_data = load_and_preprocess_dataset(dataset)
    #random_i = random.randint(0, len(image_data)-1)
    #image = image_data[random_i]["image"]
    #image = image_data[15572]["image"]
    #image = image_data[9344]["image"]

    # Step 1: Load models
    # TODO: Add more generator mllms (i.e. llava(done), molmo)
    # Setup a SLM (Llama-3.2 1B/3B) for output structure related post-processing
    slm = setup_slm()
    # generator_mllms, judge_mllm, br_mllm = setup_models(generator_models, judge_model, br_model)
    #final_judge_mllm = setup_final_judge()
    
    qar_texts = [
        """
        I apologize, but I can't provide a response to your question because I don't have enough details to solve it. The provided image does not seem to be related to the question you've asked. Please provide a new question that is relevant to the image, and I'll be happy to help you with your query.
        """,
        """
        QAR NO 1
        QUESTION: What is the man looking at?,  
        ANSWER: A pizza,  
        RATIONALE: The image shows a man holding a plate with a piece of pizza on it. The man's eyes are focused on the pizza, indicating that he is looking at it.

        QAR NO 2
        QUESTION: Is the man holding a pizza?,  
        ANSWER: Yes,  
        RATIONALE: The man is holding a plate with a piece of pizza on it.

        QAR NO 3
        QUESTION: How is the man holding the pizza?,  
        ANSWER: With his right hand,  
        RATIONALE: The man is holding the plate in his right hand, and his thumb is visible on the back of the plate.
        """,
        """
        QAR NO 1  
        QUESTION: What is the purpose of a vehicle like this in the image?,  
        ANSWER: For herding sheep.,  
        RATIONALE: A vehicle with the same color and shape as the one depicted is likely used in the image to facilitate sheep herding by guiding and controlling the sheep's movement for the purpose of farming or agriculture.

        QAR NO 2  
        QUESTION: What is the setting in the image?,  
        ANSWER: An open pasture or field,  
        RATIONALE: The image depicts a wide, open area with a large number of sheep spread out across it, surrounded by trees in the distance. This suggests a natural setting such as a farm or pasture.

        QAR NO 3  
        QUESTION: How are the sheep being managed in the image?,  
        ANSWER: Herded,  
        RATIONALE: The presence of the vehicle and the large number of sheep scattered across the field indicates that the sheep are being actively managed, possibly to move them to another location or to direct their movement.

        Answer: 
        The vehicle in the image appears to be used for herding sheep in an open pasture or field. The large number of sheep spread out across the field, accompanied by a vehicle with the same color and shape as the one depicted, suggests that the sheep are being actively managed, possibly to move them to another location or to direct their movement.
        """,
        """
        **QAR NO 1**
        QUESTION: Is the motorcycle parked legally?,  
        ANSWER: Yes,  
        RATIONALE: The motorcycle is parked in the designated parking area, facing the correct direction, and does not obstruct the path for pedestrians.

        **QAR NO 2**
        QUESTION: What is the age of the motorcycle?,  
        ANSWER: The age of the motorcycle is over 10 years old,  
        RATIONALE: The motorcycle's design and features indicate that it is an older model, with visible signs of wear and tear, such as faded paint and worn-out tires.

        **QAR NO 3**
        QUESTION: What safety feature is the rider utilizing?,  
        ANSWER: The rider is wearing a helmet,  
        RATIONALE: The rider is wearing a helmet, which is a standard safety feature for motorcycle riders, indicating their commitment to safety.
        """,
        """
        **QAR NO 1**
        **QUESTION:** What is the object in his hand?
        **ANSWER:** Briefcase.
        **RATIONALE:** The man is holding an object that is typical of a briefcase, which is used to carry items for work or other purposes.

        **QAR NO 2**
        **QUESTION:** Is the man standing behind the bus stop a tourist?
        **ANSWER:** No.
        **RATIONALE:** The man's formal clothing and posture suggest that he is not a tourist, but rather someone who is likely a local or businessperson.

        **QAR NO 3**
        **QUESTION:** What is the man looking at on the bus stop?
        **ANSWER:** A poster or advertisement.
        **RATIONALE:** The man is looking down at something on the ground near the bus stop, which appears to be a poster or advertisement.
        """,
        """
        Here are the answers to the questions based on the image:

        QAR NO 1
        QUESTION: What is the man's job?
        ANSWER: Farmer.
        RATIONALE: The man is sitting in a field, surrounded by cows and grass. The image suggests that he is taking care of the cows, possibly feeding them or performing other farm-related tasks. This is consistent with the occupation of a farmer.

        QAR NO 2
        QUESTION: Why is the cow being fed?
        ANSWER: To be nourished.
        RATIONALE: In the image, the man is holding a bucket and appears to be pouring liquid into the cow's mouth. This action is likely intended to feed the cow. Feeding a cow is essential to provide it with the necessary nutrients for survival and growth.

        QAR NO 3
        QUESTION: Can people get close to cows?
        ANSWER: Yes.
        RATIONALE: The image shows people, specifically the man, standing close to the cows. This suggests that it is possible for people to approach and interact with cows, at least in this context.
        """,
        """
        The image depicts a rainy day scene with people walking on a sidewalk, holding umbrellas. The people are dressed in dark clothing, suggesting it is either early morning or late evening.

        **QAR NO 1**
        QUESTION: Why are the people in the image holding umbrellas?, 
        ANSWER: To stay dry., 
        RATIONALE: The image shows rain falling, and the people are holding umbrellas to protect themselves from the rain.

        **QAR NO 2**
        QUESTION: How might the rain affect the car's paint job?, 
        ANSWER: It might cause the paint to rust or fade., 
        RATIONALE: The image shows a red car parked on the side of the road, and the rain is falling on it. This could potentially damage the car's paint job, causing it to rust or fade.

        **QAR NO 3**
        QUESTION: What is the man holding in his right hand?, 
        ANSWER: A bag or purse., 
        RATIONALE: The man is wearing a plaid shirt and jeans, and he has a bag or purse slung over his right shoulder.
        """
    ]

    for qar_text in qar_texts:
        print(qar_text)
        #postprocess_qars(qar_text)
        break
    
    """
    sample_syn_qars = load_json_file(file_name='sample_q_no_rules_zero_shot.json') 
    
    image_ids = []
    for syn_qar in sample_syn_qars:
        image_ids.append(syn_qar["serial"])

    generate_sample_data(image_data, 'sample_q_few_shot_cot.json', generator_mllms[0], image_ids)
    """    
    
    
    #print(random_i)
    # Remove after testing  
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)

    '''
    for idx in image_ids:
        image = image_data[idx]["image"]
    '''
    
    '''
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

        all_syn_qars = {}
        # TODO: Test with >1 mllm (i.e. llama 3.2, llava, molmo)
        for i, mllm in enumerate(generator_mllms):
            questions = mllm.generate(image, use_evol_prompt, questions=None, evolvable_questions=evolvable_questions)
            syn_qars = mllm.generate(image, use_evol_prompt=False, questions=questions, evolvable_questions=[])
            syn_qars = postprocess_qars(syn_qars)
            try:
                syn_qars = json.loads(syn_qars)
            except json.JSONDecodeError:
                print(f'Error: Could not parse syn_qars for {random_i}-th training image, moving to next mllm.')
                continue
            
            all_syn_qars[f'mllm_{i+1}'] = {}
            for j, qar in enumerate(syn_qars):
                all_syn_qars[f'mllm_{i+1}'][f'qar_{j+1}'] = qar
                    
        # Step 3: Judge (soft filter) qars generated by each generator mllm
        for mllm in all_syn_qars:
            judged_qars = []
            syn_qars = []
            for qar in all_syn_qars[mllm]:
                syn_qars.append(all_syn_qars[mllm][qar])

            judged_qars = judge_qars(syn_qars, image, judge_mllm)

            i = 0
            for qar in all_syn_qars[mllm]:
                all_syn_qars[mllm][qar] = judged_qars[i]
                i += 1
        
        # Step 4: Do inference verification using backward reasoner mllm
        for mllm in all_syn_qars:
            inference_verified_qars = []
            judged_syn_qars = []
            for qar in all_syn_qars[mllm]:
                judged_syn_qars.append(all_syn_qars[mllm][qar])
            inference_verified_qars = verify_inference(judged_syn_qars, image, br_mllm)

            i = 0
            for qar in all_syn_qars[mllm]:
                all_syn_qars[mllm][qar] = inference_verified_qars[i]
                i += 1

        # step 5: keep all filtered in quality qars
        syn_qar_bucket = []
        for mllm in all_syn_qars:
            for qar in all_syn_qars[mllm]:
                if (all_syn_qars[mllm][qar]['br_score'] > 0.7):
                    syn_qar_bucket.append(all_syn_qars[mllm][qar])
        
        # step 6: de-duplicate initially filtered qars
        unique_qars = deduplicate_qars(syn_qar_bucket)
        
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
    '''
    
    #TODO: Apply dedup again on final synthetic dataset

    #TODO: Store in huggingface



     