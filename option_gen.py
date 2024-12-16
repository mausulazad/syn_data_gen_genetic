import os
import time
import json
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

cache_dir= '/scratch/mi8uu/cache'
os.environ['TRANSFORMERS_CACHE']=cache_dir
os.environ['HF_HOME']= cache_dir

from utils import load_and_preprocess_dataset, setup_llama32, clean_out_json_output, convert_and_upload_to_hf

vlm, processor = setup_llama32()

"""
vlm_id = "microsoft/Phi-3.5-vision-instruct" 
vlm = AutoModelForCausalLM.from_pretrained(
    vlm_id, 
    #device_map="cuda",
    device_map="auto", 
    trust_remote_code=True, 
    torch_dtype="auto", 
    #_attn_implementation='flash_attention_2',
    _attn_implementation='eager',
    cache_dir=cache_dir
)

processor = AutoProcessor.from_pretrained(
    vlm_id, 
    trust_remote_code=True, 
    num_crops=16,
    cache_dir=cache_dir
)
"""

mcq_system_prompt = """You are an intelligent assistant tasked with generating multiple-choice answers for a Visual Question Answering (VQA) dataset. Your goal is to create 4 choices (including the correct answer) for each question. Ensure that the choices are thoughtfully designed to engage reasoning and critical thinking.
        Here are the detailed instructions for your task:

        ### Requirements for Generating Choices:
        1. **Correct Answer**:
            - If the correct answer is **concise**, include it directly in the list of choices.
            - If the correct answer is **lengthy**, paraphrase/rephrase it into a shorter version while maintaining its core meaning and accuracy. The shortened version should still fully answer the question.
        2. **Difficulty Distribution of Wrong Answers**:
            - One/two wrong answers should be non-trivial, requiring reasoning to eliminate.
            - The remaining wrong answer(s) can be relatively easy to discard.
        3. **Plausibility**:
            - All wrong answers should appear plausible and contextually relevant to the question and the image.
        4. **Order**:
            - Randomize the order of the choices.
            
        ### Instructions for Yes/No Questions
        When the answer is binary (Yes/No):
        1. Include "Yes" and "No" as part of the choices.
        2. Add two additional plausible but incorrect options, such as: 
          -"Maybe" or "Possibly, but unlikely."
          -"It is unclear from the image."
          -"Not specified."
        Ensure the additional options are plausible and logically related to the question and image.

        ### Input Provided:
        You will be given:
            - A **question** describing an aspect of the image.
            - The **corresponding answer** that is correct for the question.
            - The **image** to help contextualize the question and answer.

        ### Formatting Requirements:
        Provide output in the following format:

        Question: <question> 
        Correct Answer: <correct answer (paraphrased/rephrased if necessary)> 
        Choices: 
        a. <choice a> 
        b. <choice b> 
        c. <choice c> 
        d. <choice d> 
        Correct Answer: <letter of the correct choice (a, b, c, or d)>

        ### Examples:
        These examples are provided to demonstrate input-output type and structure.**Do not copy-paste the examples; instead, use them as a guide to adapt your analysis based on the provided inputs.**

        **Important Note**: These examples are not necessarily exhaustive and do not encompass all possible cases. Use the examples flexibly, adapting to the inputs provided, rather than forcing the inputs to align with the examples.
        
        **Example 1:**
        Question: What can be inferred about the car based on its appearance? 
        Correct Answer: The car is red and seems recently painted. 
        Choices:
        a. The car is red and seems recently painted.
        b. The car is old and appears scratched.
        c. The car is blue and looks brand new.
        d. The car is red but covered in dust. 
        Correct Choice: a
        
        **Example 2:**
        Question: What is the person holding in their hand? 
        Correct Answer: A book. 
        Choices:
        a. A phone.
        b. A cup.
        c. A bag.
        d. A book. 
        Correct Choice: d
        
        **Example 3:**
        Question: Where is the dog sitting? 
        Correct Answer: On the couch. 
        Choices: 
        a. On the floor. 
        b. Under the table.
        c. On the couch. 
        d. Outside in the yard. 
        Correct Choice: c
        
        **Example 4 (Rephrasing/Paraphrasing Needed):**
        Question: Why are the people gathered in the park?
        Correct Answer: The people are gathered for a community event with music, food, and activities for everyone.
        Choices:
        a. The people are exercising in the park.
        b. A community event.
        c. The people are waiting for a parade to start.
        d. The people are protesting against local policies.
        Correct Choice: b
        
        DO NOT hallucinate. DO NOT ADD FALSE THINGS TO YOUR RESPONSE"""

parser_system_prompt = """You are an assistant tasked with converting structured text containing a question, correct answer,
    choices,and the correct choice into a JSON object. Follow these instructions to format the output correctly:

    ### Input Format:
    The input will include:
    - A **Question**
    - A **Correct Answer**
    - Multiple **Choices** labeled with letters (a, b, c, d, etc.)
    - A **Correct Choice** corresponding to one of the choices.

    ### Output Format:
    Convert the input into the following JSON structure:
    ```json
    {
        "choices": [
            "<Choice 1>",
            "<Choice 2>",
            "<Choice 3>",
            "<Choice 4>"
        ],
        "correct_choice": "<The correct choice letter (a/b/c/d)>"
    }
    
    Rules:
    1. Remove the choice labels (e.g., a., b., c., d.) and list the choices in the choices array in the order they appear.
    2. Use the text corresponding to the Correct Choice field as the value for correct_choice.
    
    Notes:
    - Ensure all fields in the JSON object are correctly populated.
    - Maintain the order of choices as presented in the input.
    -The correct_choice field must match the corresponding correct choice letter (e.g., "c").
    
    Return only the JSON object. Nothing else.
    DO NOT HALLUCINATE"""

data = load_and_preprocess_dataset("Mausul/syn_dataset_no_evolution_single_run_smol_v0")

CHOICE_MAP = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3
}

def generate_options():
    total_inference_time = 0
    updated_qars = []
    for i, qar in enumerate(data):
        image = qar["image"]
        question = qar["question"]
        correct_answer = qar["answer"]
        
        start = time.time()
        messages = [
            { "role": "assistant", "content": mcq_system_prompt },
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f'Question: {question}\nCorrect Answer: {correct_answer}\nChoices:\n'}
                ]
            },
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(vlm.device)

        output = vlm.generate(
            **inputs, 
            temperature=0.3, 
            max_new_tokens=300
        )
        output = processor.decode(output[0][inputs.input_ids.shape[-1]:])
        output = output.split('<|eot_id|>')[0]
        
        messages = [
            { "role": "assistant", "content": parser_system_prompt },
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": output}
                ]
            },
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(vlm.device)

        output = vlm.generate(
            **inputs, 
            temperature=0.3, 
            max_new_tokens=300
        )
        output = processor.decode(output[0][inputs.input_ids.shape[-1]:])
        output = output.split('<|eot_id|>')[0]
        
        end = time.time()
        
        elapsed_time = end - start
        total_inference_time += elapsed_time
        
        new_fields = {"choices": [], "correct_choice_idx": None}
        output = clean_out_json_output(output)
        
        try:
            output = json.loads(output)
            qar["choices"] = output.get("choices", [])
            qar["correct_choice_idx"] = CHOICE_MAP.get(output.get("correct_choice", None), None)
        except json.JSONDecodeError:
            print(f'Error: Could not parse json object')
            qar["choices"] = []
            qar["correct_choice_idx"] = None
        
        updated_qars.append(qar)

        """
        if i % 2 == 1:
            print(f"options for {i+1} qars are generated...")
            print(f"No. of qars with options (till now): {len(updated_qars)}")
            print(f"Total inference time (till now): {total_inference_time/60:.2f} min(s)")
            print("="*80)
        """

        if i % 20 == 19:
            print(f"options for {i+1} qars are generated...")
            print(f"No. of qars with options (till now): {len(updated_qars)}")
            print(f"Total inference time (till now): {total_inference_time/60:.2f} min(s)")
            print("="*80)
        
        
    repo_name = "syn_dataset_no_evolution_single_run_smol_v0_with_choices"
    convert_and_upload_to_hf(updated_qars, repo_name, create_dataset=False)
    
if __name__ == "__main__":
    generate_options()