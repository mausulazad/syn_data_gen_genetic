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

#vlm, processor = setup_llama32()

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

mcq_system_prompt = """You are an intelligent assistant tasked with generating multiple-choice answers for a Visual Question Answering (VQA) dataset. Your goal is to create 4 concise and thoughtfully designed choices (including the correct answer) for each question. 
    ### Requirements for Generating Choices:
    1. **Correct Answer**:
      - If the correct answer is 2/3 words long, include it directly in the list of choices.
      - If the correct answer have more than 3 words, paraphrase it into a shorter version (2/3 words) that maintains its core meaning. The shortened version should still fully answer the question.
    2. **One Correct Choice**:
      - Each question must have **exactly one correct choice**. Ensure that the other choices are all incorrect but plausible.
    3. **Difficulty Distribution of Wrong Answers**:
      - Include 1/2 wrong answers that require reasoning to eliminate.
      - The remaining wrong answer(s) can be easy to discard but still plausible.
    4. **Plausibility**:
      - All wrong answers must be plausible and relevant to the question and image.
    5. **Order**:
      - Randomize the order of the choices.

    ### Instructions for Yes/No Questions:
    1. Include "Yes" and "No" as two of the choices.
    2. Add two additional plausible but incorrect options, such as:
      - "Unclear from the image."
      - "Not specified."
    3. Ensure all choices are short and logically related to the question and image.

    ### Formatting Requirements:
    Provide output in the following format:

    **Question**: <question>  
    **Correct Answer**: <correct answer (paraphrased/shortened if necessary)>  
    **Choices**:  
    a. <choice a>  
    b. <choice b>  
    c. <choice c>  
    d. <choice d>  
    **Correct Choice**: <letter of the correct choice (a, b, c, or d)>

    ### Updated Examples for Conciseness:

    **Example 1**:  
    **Question**: What color is the car?  
    **Correct Answer**: Red.  
    **Choices**:  
    a. Blue  
    b. Red  
    c. Black  
    d. Green  
    **Correct Choice**: b  

    **Example 2**:  
    **Question**: What is the person holding?  
    **Correct Answer**: A book.  
    **Choices**:  
    a. A cup  
    b. A phone  
    c. A book  
    d. A bag  
    **Correct Choice**: c  

    **Example 3**:  
    **Question**: Where is the dog sitting?  
    **Correct Answer**: On the couch.  
    **Choices**:  
    a. floor  
    b. couch  
    c. Under table  
    d. In the yard  
    **Correct Choice**: b  

    **Example 4 (Paraphrasing Needed)**:  
    **Question**: Why are the people in the park?  
    **Correct Answer**: For a community event.  
    **Choices**:  
    a. Exercising  
    b. A community event  
    c. Waiting for parade  
    d. Protesting  
    **Correct Choice**: b  

    **Key Emphasis**: Keep all choices concise (2/3 words) and contextually relevant while adhering to the rules."""

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

CHOICE_MAP = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3
}

def generate_options(image, question, answer):
    correct_answer = answer

    start = time.time()
    messages = [
        { "role": "assistant", "content": mcq_system_prompt },
        {
            "role": "user", 
            "content": [
                { "type": "image" },
                { "type": "text", "text": f'Question: {question}\nCorrect Answer: {correct_answer}\nChoices:\n' }
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
        temperature=0.7, 
        max_new_tokens=300
    )
    output = processor.decode(output[0][inputs.input_ids.shape[-1]:])
    output = output.split('<|eot_id|>')[0]
        
    messages = [
        { "role": "assistant", "content": parser_system_prompt },
        {
            "role": "user", 
            "content": [
                { "type": "image" },
                { "type": "text", "text": output}
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
        temperature=0.7, 
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
        choices = output.get("choices", [])
        correct_choice = output.get("correct_choice", None)
        if isinstance(correct_choice, list):
            if len(correct_choice) == 1:
                correct_choice = correct_choice[0]
            else:
                print(f"Warning: `correct_choice` is a list with length {len(correct_choice)}. Skipping this entry.")
                correct_choice = None
        correct_choice_idx = CHOICE_MAP.get(correct_choice, None)
    except json.JSONDecodeError:
        print(f'Error: Could not parse json object')
        choices = []
        correct_choice_idx = None
    
    return (choices, correct_choice_idx)


"""
def generate_options(qars):
    #qars = load_and_preprocess_dataset("Mausul/syn_dataset_no_evolution_single_run_smol_v3")
    total_inference_time = 0
    updated_qars = []
    for i, qar in enumerate(qars):
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
            temperature=0.7, 
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
            temperature=0.7, 
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
            correct_choice = output.get("correct_choice", None)
            if isinstance(correct_choice, list):
                if len(correct_choice) == 1:
                    correct_choice = correct_choice[0]
                else:
                    print(f"Warning: `correct_choice` is a list with length {len(correct_choice)}. Skipping this entry.")
                    correct_choice = None
            qar["correct_choice_idx"] = CHOICE_MAP.get(correct_choice, None)
        except json.JSONDecodeError:
            print(f'Error: Could not parse json object')
            qar["choices"] = []
            qar["correct_choice_idx"] = None
        
        updated_qars.append(qar)

        if i % 2 == 1:
            print(f"options for {i+1} qars are generated...")
            print(f"No. of qars with options (till now): {len(updated_qars)}")
            print(f"Total inference time (till now): {total_inference_time/60:.2f} min(s)")
            print("="*80)

        if i % 20 == 19:
            print(f"options for {i+1} qars are generated...")
            print(f"No. of qars with options (till now): {len(updated_qars)}")
            print(f"Total inference time (till now): {total_inference_time/60:.2f} min(s)")
            print("="*80)
        
        if i >= 8:
            break
        
    repo_name = "test_op_gen"
    #repo_name = "syn_dataset_no_evolution_single_run_smol_v3_with_choices"
    convert_and_upload_to_hf(updated_qars, repo_name, create_dataset=False)
    
if __name__ == "__main__":
    generate_options()
"""