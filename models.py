import re
import json
import random

import ast
import pprint
import copy

import torch
from transformers import GenerationConfig

from sentence_transformers import SentenceTransformer, util

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import pipeline

GENERATION_CRITERIA = [
    "The question should require commonsense knowledge about human social behavior to answer.",
    "The question should require knowledge of the physical world to answer.",
    "The question should necessitate visual understanding to answer.",
    "The question should challenge the system's reasoning capabilities.",
    "The question should be sufficiently complex to require in-depth reasoning."
]

EVOLUTION_CRITERIA = [
    "Questions should require reasoning that goes beyond simple observation."
    "Questions should focus on 'when', 'where', and 'what' aspects, and only use 'how' for answers that are concise (at max. four words)."
    "Ensure questions are grounded in the original question and the evolution method without hallucinating details."
    "Questions must integrate human social behavior, physical world knowledge, or reasoning capabilities."
    "Questions should be diverse, non-trivial, and emphasize advanced reasoning."
]

FEW_SHOT_QUESTIONS = [
    "What might happen if the person in the image dropped the object they are holding?",
    "How might the people in the image respond if one of them started laughing?",
    "Based on the setting in the image, what time of day is most likely depicted?",
    "What might the person in the image do next based on their posture?",
    "How could the objects in the image interact with one another?"
]

FEW_SHOT_EVOLUTIONS = [
    "Original Question: What is the weather like in the image?\nEvolution Method: To add complexity, ask about the time of year based on visual cues.\nEvolved Question: What season does the weather in the image suggest?",
    "Original Question: How many suitcases does the man have?\nEvolution Method: Add reasoning by focusing on the purpose of the suitcases.\nEvolved Question: What does the number of suitcases suggest about the man's trip?",
    "Original Question: Where is the woman in the image going?\nEvolution Method: Incorporate clues from the surroundings to specify the destination.\nEvolved Question: What type of location is the woman likely heading to, based on the setting?",
    "Original Question: What time of day is it in the image?\nEvolution Method: Add more context or details to increase complexity.\nEvolved Question: When does the time of day in the image suggest the scene might occur?"
]



class MLLM:
    def __init__(self, model, processor, model_family, inference_type):
        self.model = model
        self.processor = processor
        self.model_family = model_family
        self.inference_type = inference_type
        
        # TODO: Randomize and build system prompt using these two inside 'generate' method (do not update original system prompt though)
        self.generation_criteria = GENERATION_CRITERIA
        self.evolution_criteria = EVOLUTION_CRITERIA
        self.few_shot_questions = FEW_SHOT_QUESTIONS
        self.few_shot_evolutions = FEW_SHOT_EVOLUTIONS
        if inference_type == "generate": 
            self.question_system_prompt = self.get_system_prompt(inference_type, generate_questions=True)
            self.evolution_system_prompt = self.get_system_prompt(inference_type, generate_questions=True, evol_prompt=True)
            self.answer_system_prompt = self.get_system_prompt(inference_type, generate_questions=False)
        # TODO: separate system prompt out --> verification/initial_judgement
        else:
            self.system_prompt = self.get_system_prompt(inference_type, generate_questions=False)

    def get_system_prompt(self, inference_type, generate_questions=True, evol_prompt=False):
        prompt = None
        if inference_type == "generate":
            if generate_questions:
                if not evol_prompt:
                    prompt = """You are an helpful assistant. Based on a given image, you will generate questions that challenge advanced reasoning systems like ChatGPT or GPT-4. Ensure the generated questions adhere to the following crieteria:
                        
                        ###CRITERIA###:
                        {criteria}

                        When generating questions, follow a chain-of-thought approach:
                        - First, carefully analyze the image content to identify key visual elements, interactions, or context.
                        - Next, consider how these elements relate to human social behavior, physical world knowledge, and reasoning capabilities.
                        - Then, formulate a question that integrates these observations, ensuring it is relevant to the image and adheres to at least one of the criteria.
                        - Finally, verify the question for diversity, non-triviality, and adherence to the task requirements.

                        Here are few examples of such questions:
                        {examples}

                        Important instructions:
                        - DO NOT directly copy-paste these example questions, as they are not directly based on the given image. Use them as inspiration to create new, unique questions specifically relevant to the given image.
                        - NEVER HALLUCINATE information or details that are not present or cannot be inferred from the given image. Ensure all generated questions are grounded in the image content.

                        Strive to generate questions that meet as many of these criteria as possible."""
                else:
                    prompt = """You are a helpful assistant. Based on a previously generated question and its corresponding evolution method, you will generate a follow-up question that challenges advanced reasoning systems like ChatGPT or GPT-4. Ensure the follow-up question adheres to the following CRITERIA:
                    
                        ###CRITERIA###:
                        {criteria}

                        When generating follow-up question, follow this chain-of-thought approach:
                        -First, analyze the original question: Understand its context, complexity, and reasoning focus.
                        -Next, assess the evolution method; identify how the original question can be refined, e.g., by adding details, shifting focus, or integrating more dimensions.
                        -Then, formulate the follow-up question that:
                        a. Builds on the original question's theme or complexity.
                        b. Incorporates elements from the evolution method to enhance diversity and alignment.
                        c. Challenges advanced reasoning in a meaningful way.
                        -Finally, verify the question; ensure it meets diversity, non-triviality, and task requirements.
                    
                        Here are few examples of such evolutions:
                        {examples}
                    
                        Important instructions:
                        - DO NOT directly copy-paste these examples, as they are not directly based on the given image. Use them as inspiration to evolve given question to a new question that is relevant to the given image.
                        - NEVER HALLUCINATE information or details that are not present or cannot be inferred from the given image. Ensure the evolved question is grounded in the image content.

                        Strive to evolve question that meet as many of these criteria as possible."""
            else:
                prompt = """You are an helpful assistant. Based on a given image and questions (related to that image) you answer the questions by providing correct answers with justifications. To maintain high-quality and accurate responses, adhere to the following guidelines:

                    ###GUIDELINES###:
                    1. Ensure all answers are grounded in the image's content. Internal knowledge may also be used to enhance answers, but only when it aligns logically with the image and does not involve hallucination.
                    2. Follow a chain-of-thought reasoning process internally:
                        - First, analyze the image to identify relevant details and context.
                        - Then, interpret the questions in relation to these details.
                        - Formulate concise answers based on observable evidence in the image, supplemented with internal knowledge if relevant.
                        - Provide brief rationales that logically justify why the answers are correct.
                    3. When reasoning, focus on visual clues such as actions, objects, expressions, and environmental factors, complemented by valid external knowledge if it aids in providing better answers.
                    4. Keep all responses precise and focused, avoiding over-explanation.
                    5. Do not directly copy-paste any provided examples, as they are unrelated to the given image. The examples are intended solely for understanding tone, format, and quality expectations.
                    6. Return answers and rationales in a structured format, ensuring all triplets are included in a single list of JSON objects.
                    7. Note: An actual image (not an image description) will be provided as input. Use the visual content of the image, along with relevant internal knowledge, to formulate your responses.

                    ###EXAMPLES###:
                        
                    ###EXAMPLE 1:
                    #### Example Input:
                    Image: [An image showing a woman holding an umbrella on a rainy street with other people walking by.]
                    Questions: ["Why is the woman holding the umbrella?", "What might happen if the woman closes the umbrella?"]

                    #### Example Output:
                    QAR NO 1  
                    QUESTION: Why is the woman holding the umbrella?,  
                    ANSWER: To stay dry.,  
                    RATIONALE: The image shows rain falling and the woman holding the umbrella open. Using an umbrella in rain is a common practice to stay dry.  
                            
                    QAR NO 2
                    QUESTION: What might happen if the woman closes the umbrella?,  
                    ANSWER: She will get dranched.,  
                    RATIONALE: The image depicts rain, and an umbrella is used to shield from it. Closing the umbrella would expose her to the rain.

                    ###EXAMPLE 2:
                    #### Example Input:
                    Image: [An image showing a dog lying under a tree on a sunny day, with a ball placed nearby.]
                    Questions: ["Why is the dog lying under the tree?", "Why is the ball not being played with?"]
        
                    #### Example Output:
                    QAR NO 1  
                    QUESTION: Why is the dog lying under the tree?,  
                    ANSWER: For shade.,  
                    RATIONALE: The image shows it is sunny, and the tree provides shade. Dogs often seek shade to avoid direct sunlight and cool down.  

                    QAR NO 2  
                    QUESTION: Why is the ball not being played with?,  
                    ANSWER: The dog is resting.,  
                    RATIONALE: The dog appears to be lying down and resting under the tree, suggesting it is not active at the moment to play with the ball.

                    Use these examples as a reference for tone, style, format, and quality level. Ensure all question-answer-rationale triplets follow this specified format and are preceded by the `QAR NO <number>` separator. DO NOT directly copy or adapt these examples. DO NOT include any intermediate reasoning, CoT steps, or explanations in the output.
                            
                    Strive to answer questions with proper rationale that are on-par with the questions (quality-wise)."""
        # TODO: initial judge and backward reasoner
        else:
            prompt = None

        return prompt

    def randomize_criteria(self, criteria):
        randomized_criteria = random.sample(criteria, len(criteria))
        #random.shuffle(self.criteria)
        randomized_criteria = [f"{i+1}. {criterion}" for i, criterion in enumerate(randomized_criteria)]
        randomized_criteria = "\n".join(randomized_criteria)
        return randomized_criteria

    def randomized_few_shot_questions(self, examples, type):
        randomized_few_shot_questions = random.sample(examples, len(examples))
        if type == "non_evolve":
            #random.shuffle(self.few_shot_questions)
            randomized_few_shot_questions = [f"Example {i+1}: {question}" for i, question in enumerate(randomized_few_shot_questions)]
        randomized_few_shot_questions = "\n".join(randomized_few_shot_questions)
        return randomized_few_shot_questions
    
    def generate_using_llama32(self, image, use_evol_prompt, questions, evolvable_questions, criteria, few_shot_examples, max_new_tokens):
        if self.inference_type == "generate":
            # Generating questions
            if questions is None:
                # Evolving generated questions
                if use_evol_prompt:
                    input_str = ''
                    for i, evolvable_question in enumerate(evolvable_questions):
                        input_str += f'Original Question {i+1}: {evolvable_question[0]}\nEvolution method {i+1}: {evolvable_question[1]}\n'
                
                    
                    query = """
                        1. Generate exactly {count_of_evolvable_questions} diverse and non-trivial questions based on the given image, given questions and corresponding evolution methods.
                        2. When generating follow-up questions, follow this chain-of-thought approach:
                          -First, analyze the original question: Understand its context, complexity, and reasoning focus.
                          -Next, assess the evolution method; identify how the original question can be refined, e.g., by adding details, shifting focus, or integrating more dimensions.
                          -Then, formulate the follow-up question that:
                            a. Builds on the original question's theme or complexity.
                            b. Incorporates elements from the evolution method to enhance diversity and alignment.
                            c. Challenges advanced reasoning in a meaningful way.
                          -Finally, verify the question; ensure it meets diversity, non-triviality, and task requirements.
                        3. Ensure all evolved questions are strictly based on the content of the image and do not introduce details or concepts that cannot be directly inferred.
                        4. DO NOT hallucinate any information; all questions must be fully grounded in the given image.
                        5. Follow the rules outlined in the system prompt, ensuring relevance, clarity, and proper formatting of the questions.
                        6. Each question must:
                            - End with a '?'.
                            - Be answerable in 1 to 5 words only.
                            - Not include any sub-questions.
                        7. Return the questions as a comma-separated list enclosed in square brackets, formatted like this: [<question 1>, <question 2>, <question 3>, ...].
                        8. DO NOT provide answers or any additional information beyond the formatted list.
                        9. Ensure that there are no fewer than {count_of_evolvable_questions} questions, and all strictly follow these rules.
                        
                        Inputs:
                        {input_str}"""

                    query = query.format(count_of_evolvable_questions=len(evolvable_questions), input_str=input_str)
                    system_prompt = self.evolution_system_prompt.format(criteria=criteria, examples=few_shot_examples)      
                # 1st time generating questions
                else:
                    query = """
                        1. Generate exactly 3 diverse and non-trivial questions based on the given image.
                        2. Use a chain-of-thought approach to generate questions:
                            - Analyze the image to identify key visual elements, interactions, or context.
                            - Consider how these elements relate to human social behavior, physical world knowledge, and reasoning capabilities.
                            - Formulate a question that integrates these observations, ensuring it is relevant to the image and adheres to at least one of the criteria.
                            - Verify the question for diversity, non-triviality, and adherence to the task requirements.
                        3. Ensure all questions are strictly based on the content of the image and do not introduce details or concepts that cannot be directly inferred.
                        4. DO NOT hallucinate any information; all questions must be fully grounded in the given image.
                        5. Follow the rules outlined in the system prompt, ensuring relevance, clarity, and proper formatting of the questions.
                        6. Each question must:
                            - End with a '?'.
                            - Be answerable in 1 to 5 words only.
                            - Not include any sub-questions.
                        7. Return the questions as a comma-separated list enclosed in square brackets, formatted like this: [<question 1>, <question 2>, <question 3>].
                        8. DO NOT provide answers or any additional information beyond the formatted list.
                        9. Ensure that there are no fewer than 3 questions, and all strictly follow these rules."""
                
                    system_prompt = self.question_system_prompt.format(criteria=criteria, examples=few_shot_examples)
                
                messages = [
                    { "role": "assistant", "content": system_prompt },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": query}
                        ]
                    },
                ]
            # Generating answers and rationales
            else:
                query = """You will be given an image and questions based on the image. For each question:
                    1. Use a chain-of-thought reasoning style internally for answer and rationale generation:
                        - First, analyze the image to identify relevant details and context.
                        - Then, interpret the question in relation to these details.
                        - Formulate concise answers based on observable evidence in the image, supplemented with internal knowledge if relevant.
                        - Provide brief rationales that logically justify why the answers are correct.
                    2. Generate a correct and precise answer based on the content of the image. You may also use internal knowledge to aid in answering the question, but only if the knowledge aligns logically with the image content and does not involve hallucination.
                    3. Provide a corresponding brief rationale to justify why the answer is correct.
                    4. Ensure that both the answer and rationale are grounded in observable details from the image, supplemented by valid knowledge where applicable.
                    5. Keep the answers and rationales concise, avoiding over-explanation.

                    Important Instructions:
                    - DO NOT directly copy-paste any provided examples, as they are unrelated to the given image. Examples are solely for understanding tone, style, format, and quality level expected in your responses.
                    - NEVER HALLUCINATE information or provide details that are not supported by the image or relevant, valid knowledge.

                    Precede each triplet (QUESTION, ANSWER, RATIONALE) with the tag `QAR NO <number>`. Use the following format:
                    
                    QAR NO <number>  
                    QUESTION: <question text>,  
                    ANSWER: <corresponding correct answer>,  
                    RATIONALE: <corresponding rationale>.  

                    Instead of an image description, an actual image will be provided. Use its content along with valid internal knowledge to formulate your responses.

                    DO NOT include any intermediate reasoning, CoT steps, or explanations in the output. Return only the structured response in the specified format, nothing else."""
                
                messages = [
                    { "role": "assistant", "content": self.answer_system_prompt },
                    { "role": "assistant", "content": f"Questions: {questions}" },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": query}
                        ]
                    },
                ]
        elif self.inference_type == "backward_reasoning":
            messages = [
                { "role": "assistant", "content": self.system_prompt },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f'Here is the answer-rationale pair: {questions}'}
                    ]
                },
            ]
        elif self.inference_type == "judge":
            messages = [
                { "role": "assistant", "content": self.system_prompt },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f'Here is the qar: {questions}'}
                    ]
                },
            ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs, 
            temperature=1.0, 
            max_new_tokens=max_new_tokens
        )
        output = self.processor.decode(output[0][inputs.input_ids.shape[-1]:])
        output = output.split('<|eot_id|>')[0]
        return output

    def generate_using_llava(self, image, use_evol_prompt, questions, evolvable_questions):
        if questions is None:
            if use_evol_prompt:
                pass
            else:

                query = "Generate 3 non-trivial, diverse questions (add '?' after each question) based on the image without hallucinating that can be answered using one to at max. five words. DO NOT ANSWER, JUST GENERATE QUESTIONS. Each question can not have sub-questions. Return all questions inside a list (comma separated) like this: [<question 1>, <question 2>, <question 3>]. Return only the list of questions (generate 3 questions, NOT LESS THAN THAT. AND MUST RETURN THEM INSIDE A LIST.), nothing else."
                messages = [
                    f"ASSISTANT: {self.system_prompt}\nUSER: <image>\n{query}\nASSISTANT:"
                ]
        else:
            #query = 'You will be given an image and questions that are based on the image. For each question generate correct answer and corresponding brief rationale (rationale should justify briefly why the answer is correct, give proper reasoning) without hallucinating. Keep the answers precise and short (no over explanation). Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. YOU MUST RETURN ALL NON-DUPLICATE JSON OBJECTS INSIDE A LIST.Return only the list of JSON objects, nothing else.'
            query = 'Given an image and questions based on the image, generate a correct answer and a corresponding brief but insightful rationale for each question. The rationale should justify why the answer is correct by referencing specific details in the image or using logical inference when appropriate. Avoid vague statements; each rationale should clarify how the visible elements or context in the image supports the answer. Ensure answers are precise and concise (1-2 words if possible), and the rationale directly connects to the answer without over-explanation. Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. YOU MUST RETURN ALL NON-DUPLICATE JSON OBJECTS INSIDE A LIST.Return only the list of JSON objects, nothing else.'
            messages = [
                f"ASSISTANT: {self.system_prompt}\nUSER: <image>\nHere are the questions: {questions}\n{query}\nASSISTANT:"
            ]
        
        inputs = self.processor(
            image,
            messages,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        
        #inputs = self.processor(messages, images=[image], padding=True, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=300)
        output = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        output = output.split("\nASSISTANT: ")[1]
        return output


    def generate_using_llava_next(self, image, use_evol_prompt, questions, evolvable_questions, criteria, few_shot_examples, max_new_tokens):
        # Generating questions
        if questions is None:
            # Evolving generated questions
            if use_evol_prompt:
                input_str = ''
                for i, evolvable_question in enumerate(evolvable_questions):
                    input_str += f'Original Question {i+1}: {evolvable_question[0]}\nEvolution method {i+1}: {evolvable_question[1]}\n'
                
                query = """
                    1. Generate exactly {count_of_evolvable_questions} diverse and non-trivial questions based on the given image, given questions and corresponding evolution methods.
                    2. When generating follow-up questions, follow this chain-of-thought approach:
                        -First, analyze the original question: Understand its context, complexity, and reasoning focus.
                        -Next, assess the evolution method; identify how the original question can be refined, e.g., by adding details, shifting focus, or integrating more dimensions.
                        -Then, formulate the follow-up question that:
                            a. Builds on the original question's theme or complexity.
                            b. Incorporates elements from the evolution method to enhance diversity and alignment.
                            c. Challenges advanced reasoning in a meaningful way.
                        -Finally, verify the question; ensure it meets diversity, non-triviality, and task requirements.
                    3. Ensure all evolved questions are strictly based on the content of the image and do not introduce details or concepts that cannot be directly inferred.
                    4. DO NOT hallucinate any information; all questions must be fully grounded in the given image.
                    5. Follow the rules outlined in the system prompt, ensuring relevance, clarity, and proper formatting of the questions.
                    6. Each question must:
                        - End with a '?'.
                        - Be answerable in 1 to 5 words only.
                        - Not include any sub-questions.
                    7. Return the questions as a comma-separated list enclosed in square brackets, formatted like this: [<question 1>, <question 2>, <question 3>, ...].
                    8. DO NOT provide answers or any additional information beyond the formatted list.
                    9. Ensure that there are no fewer than {count_of_evolvable_questions} questions, and all strictly follow these rules.
                        
                    Inputs:
                    {input_str}"""

                query = query.format(count_of_evolvable_questions=len(evolvable_questions), input_str=input_str)
                system_prompt = self.evolution_system_prompt.format(criteria=criteria, examples=few_shot_examples)         
            # 1st time generating questions
            else:
                query = """
                    1. Generate exactly 3 diverse and non-trivial questions based on the given image.
                    2. Use a chain-of-thought approach to generate questions:
                        - Analyze the image to identify key visual elements, interactions, or context.
                        - Consider how these elements relate to human social behavior, physical world knowledge, and reasoning capabilities.
                        - Formulate a question that integrates these observations, ensuring it is relevant to the image and adheres to at least one of the criteria.
                        - Verify the question for diversity, non-triviality, and adherence to the task requirements.
                    3. Ensure all questions are strictly based on the content of the image and do not introduce details or concepts that cannot be directly inferred.
                    4. DO NOT hallucinate any information; all questions must be fully grounded in the given image.
                    5. Follow the rules outlined in the system prompt, ensuring relevance, clarity, and proper formatting of the questions.
                    6. Each question must:
                        - End with a '?'.
                        - Be answerable in 1 to 5 words only.
                        - Not include any sub-questions.
                    7. Return the questions as a comma-separated list enclosed in square brackets, formatted like this: [<question 1>, <question 2>, <question 3>].
                    8. DO NOT provide answers or any additional information beyond the formatted list.
                    9. Ensure that there are no fewer than 3 questions, and all strictly follow these rules."""
                
                system_prompt = self.question_system_prompt.format(criteria=criteria, examples=few_shot_examples)
            
            messages = [
                { 
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ] 
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                },
            ]
        # Generating answers and rationales
        else:
            query = """You will be given an image and questions based on the image. For each question:
                1. Use a chain-of-thought reasoning style internally for answer and rationale generation:
                    - First, analyze the image to identify relevant details and context.
                    - Then, interpret the question in relation to these details.
                    - Formulate concise answers based on observable evidence in the image, supplemented with internal knowledge if relevant.
                    - Provide brief rationales that logically justify why the answers are correct.
                2. Generate a correct and precise answer based on the content of the image. You may also use internal knowledge to aid in answering the question, but only if the knowledge aligns logically with the image content and does not involve hallucination.
                3. Provide a corresponding brief rationale to justify why the answer is correct.
                4. Ensure that both the answer and rationale are grounded in observable details from the image, supplemented by valid knowledge where applicable.
                5. Keep the answers and rationales concise, avoiding over-explanation.

                Important Instructions:
                - DO NOT directly copy-paste any provided examples, as they are unrelated to the given image. Examples are solely for understanding tone, style, format, and quality level expected in your responses.
                - NEVER HALLUCINATE information or provide details that are not supported by the image or relevant, valid knowledge.

                Precede each triplet (QUESTION, ANSWER, RATIONALE) with the tag `QAR NO <number>`. Use the following format:
                    
                QAR NO <number>  
                QUESTION: <question text>,  
                ANSWER: <corresponding correct answer>,  
                RATIONALE: <corresponding rationale>.  

                Instead of an image description, an actual image will be provided. Use its content along with valid internal knowledge to formulate your responses.

                DO NOT include any intermediate reasoning, CoT steps, or explanations in the output. Return only the structured response in the specified format, nothing else."""
            
            messages = [
                { 
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": self.answer_system_prompt}
                    ]
                },
                { 
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": questions}
                    ] 
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                },
            ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            return_tensors="pt"
        ).to(self.model.device)

        # TODO: Set same temperature for all generator MLLMs
        #output = self.model.generate(**inputs, temperature=0.3, max_new_tokens=150)
        output = self.model.generate(
            **inputs,
            temperature=1.0, 
            max_new_tokens=max_new_tokens
        )
        output = self.processor.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return output

    def generate_using_phi3(self, image, use_evol_prompt, questions, evolvable_questions):
        placeholder = ""
        placeholder += f"<|image_1|>\n"

        if self.inference_type == "generate":
            if questions is None:
                if use_evol_prompt:
                    input_str = ''
                    for evolvable_question in evolvable_questions:
                        input_str += f'Original Question: {evolvable_question["question"]} Evolution Strategy: {evolvable_question["evolution_inst"]}\n'
                
                    # use evol method along with past questions for evolution (use evolvable_questions)
                    query = ("Given the following list of original questions and their corresponding evolution strategies, improve each original " 
                        "question by following its specific evolution strategy. Ensure that each evolved question requires commonsense knowledge, " 
                        "understanding of the physical world, reasoning capabilities, and/or in-depth complexity, and that it can still be answered " 
                        "in one to five words. Each evolved question should not have sub-questions and must retain relevance to the context of the image "
                        "and must NOT copy-paste or use the example questions from the evolution strategies directly.\n\n" 
                        "Inputs:\n"
                        f"{input_str}\n"
                        "Return the evolved questions as a list, comma separated and in one line, like this: [<evolved_question_1>, <evolved_question_2>, <evolved_question_3>], nothing else.")
                else:
                    query = "Generate 3 non-trivial, diverse questions (add '?' after each question) based on the image without hallucinating that can be answered using one to at max. five words. Each question can not have sub-questions. Return the questions in a list (comma separated) like this: [<question 1>, <question 2>, <question 3>]. Return only the list of questions, nothing else."
                messages = [
                    {"role": "assistant", "content": self.system_prompt},
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": query}
                ]
            else:
                query = 'You will be given an image and questions that are based on the image. For each question generate correct answer and corresponding brief rationale (rationale should justify briefly why the answer is correct) without hallucinating. Keep the answers precise and short (no over explanation). Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. Return only the list of JSON objects, nothing else.'
                messages = [
                    {"role": "assistant", "content": self.system_prompt},
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": query},
                    {"role": "user", "content": f'Here are the questions: {questions}'}
                ]
        elif self.inference_type == "backward_reasoning":
            messages = [
                {"role": "assistant", "content": self.system_prompt},
                {"role": "user", "content": placeholder},
                {"role": "user", "content": f'Here is the answer-rationale pair: {questions}'}
            ]
        elif self.inference_type == "judge":
            messages = [
                {"role": "assistant", "content": self.system_prompt},
                {"role": "user", "content": placeholder},
                {"role": "user", "content": f'Here is the qar: {questions}'}
            ]
        
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt").to(self.model.device) 

        generation_args = { 
            "max_new_tokens": 300, 
            "temperature": 0.3, 
            "do_sample": False,
        }

        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        output = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0] 
        return output

    def generate_using_molmo(self, image, use_evol_prompt, questions, evolvable_questions, criteria, few_shot_examples, max_new_tokens):
        if questions is None:
            if use_evol_prompt:
                input_str = ''
                for i, evolvable_question in enumerate(evolvable_questions):
                    input_str += f'Original Question {i+1}: {evolvable_question[0]}\nEvolution method {i+1}: {evolvable_question[1]}\n'
                
                query = """
                    1. Generate exactly {count_of_evolvable_questions} diverse and non-trivial questions based on the given image, given questions and corresponding evolution methods.
                    2. When generating follow-up questions, follow this chain-of-thought approach:
                        -First, analyze the original question: Understand its context, complexity, and reasoning focus.
                        -Next, assess the evolution method; identify how the original question can be refined, e.g., by adding details, shifting focus, or integrating more dimensions.
                        -Then, formulate the follow-up question that:
                            a. Builds on the original question's theme or complexity.
                            b. Incorporates elements from the evolution method to enhance diversity and alignment.
                            c. Challenges advanced reasoning in a meaningful way.
                        -Finally, verify the question; ensure it meets diversity, non-triviality, and task requirements.
                    3. Ensure all evolved questions are strictly based on the content of the image and do not introduce details or concepts that cannot be directly inferred.
                    4. DO NOT hallucinate any information; all questions must be fully grounded in the given image.
                    5. Follow the rules outlined in the system prompt, ensuring relevance, clarity, and proper formatting of the questions.
                    6. Each question must:
                        - End with a '?'.
                        - Be answerable in 1 to 5 words only.
                        - Not include any sub-questions.
                    7. Return the questions as a comma-separated list enclosed in square brackets, formatted like this: [<question 1>, <question 2>, <question 3>, ...].
                    8. DO NOT provide answers or any additional information beyond the formatted list.
                    9. Ensure that there are no fewer than {count_of_evolvable_questions} questions, and all strictly follow these rules.
                        
                    Inputs:
                    {input_str}"""

                query = query.format(count_of_evolvable_questions=len(evolvable_questions), input_str=input_str)
                system_prompt = self.evolution_system_prompt.format(criteria=criteria, examples=few_shot_examples) 
            # Generating questions for the 1st time
            else:
                query = """
                    1. Generate exactly 3 diverse and non-trivial questions based on the given image.
                    2. Use a chain-of-thought approach to generate questions:
                        - Analyze the image to identify key visual elements, interactions, or context.
                        - Consider how these elements relate to human social behavior, physical world knowledge, and reasoning capabilities.
                        - Formulate a question that integrates these observations, ensuring it is relevant to the image and adheres to at least one of the criteria.
                        - Verify the question for diversity, non-triviality, and adherence to the task requirements.
                    3. Ensure all questions are strictly based on the content of the image and do not introduce details or concepts that cannot be directly inferred.
                    4. DO NOT hallucinate any information; all questions must be fully grounded in the given image.
                    5. Follow the rules outlined in the system prompt, ensuring relevance, clarity, and proper formatting of the questions.
                    6. Each question must:
                        - End with a '?'.
                        - Be answerable in 1 to 5 words only.
                        - Not include any sub-questions.
                    7. Return the questions as a comma-separated list enclosed in square brackets, formatted like this: [<question 1>, <question 2>, <question 3>].
                    8. DO NOT provide answers or any additional information beyond the formatted list.
                    9. Ensure that there are no fewer than 3 questions, and all strictly follow these rules."""

                system_prompt = self.question_system_prompt.format(criteria=criteria, examples=few_shot_examples)
        # Generating answers and rationales
        else:
            query = """You will be given an image and questions based on the image. For each question:
                1. Use a chain-of-thought reasoning style internally for answer and rationale generation:
                    - First, analyze the image to identify relevant details and context.
                    - Then, interpret the question in relation to these details.
                    - Formulate concise answers based on observable evidence in the image, supplemented with internal knowledge if relevant.
                    - Provide brief rationales that logically justify why the answers are correct.
                2. Generate a correct and precise answer based on the content of the image. You may also use internal knowledge to aid in answering the question, but only if the knowledge aligns logically with the image content and does not involve hallucination.
                3. Provide a corresponding brief rationale to justify why the answer is correct.
                4. Ensure that both the answer and rationale are grounded in observable details from the image, supplemented by valid knowledge where applicable.
                5. Keep the answers and rationales concise, avoiding over-explanation.

                Important Instructions:
                - DO NOT directly copy-paste any provided examples, as they are unrelated to the given image. Examples are solely for understanding tone, style, format, and quality level expected in your responses.
                - NEVER HALLUCINATE information or provide details that are not supported by the image or relevant, valid knowledge.

                Precede each triplet (QUESTION, ANSWER, RATIONALE) with the tag `QAR NO <number>`. Use the following format:
                    
                QAR NO <number>  
                QUESTION: <question text>,  
                ANSWER: <corresponding correct answer>,  
                RATIONALE: <corresponding rationale>.  

                Instead of an image description, an actual image will be provided. Use its content along with valid internal knowledge to formulate your responses.

                DO NOT include any intermediate reasoning, CoT steps, or explanations in the output. Return only the structured response in the specified format, nothing else."""
            
            system_prompt = self.answer_system_prompt
        
        prompt = None
        if questions is None:
            prompt = f'{system_prompt}\n\n{query}'
        else:
            prompt = f'{system_prompt}\n\n{query}\n\nQuestions: {questions}'

        inputs = self.processor.process(
            images=[image],
            text=prompt
        )

        #inputs["images"] = inputs["images"].to(torch.bfloat16)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=300, temperature=1.0, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        generated_tokens = output[0,inputs['input_ids'].size(1):]
        output = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output

    # base-64 encoded image
    def generate(self, image, use_evol_prompt=False, questions=None, evolvable_questions=[], max_new_tokens=300):
        output = None
        if not use_evol_prompt:
            randomized_criteria = self.randomize_criteria(self.generation_criteria)
            randomized_few_shot_examples = self.randomized_few_shot_questions(self.few_shot_questions, type="non_evolve")
        #TODO
        else:
            randomized_criteria = self.randomize_criteria(self.evolution_criteria)
            randomized_few_shot_examples = self.randomized_few_shot_questions(self.few_shot_evolutions, type="evolve")
            
        if self.model_family == "llama_32":
            output = self.generate_using_llama32(image, use_evol_prompt, questions, evolvable_questions, randomized_criteria, randomized_few_shot_examples, max_new_tokens)
        elif self.model_family == "llava_next":
            output = self.generate_using_llava_next(image, use_evol_prompt, questions, evolvable_questions, randomized_criteria, randomized_few_shot_examples, max_new_tokens)
        elif self.model_family == "molmo":
            output = self.generate_using_molmo(image, use_evol_prompt, questions, evolvable_questions, randomized_criteria, randomized_few_shot_examples, max_new_tokens)
        '''
        elif self.model_family == "llava":
            output = self.generate_using_llava(image, use_evol_prompt, questions, evolvable_questions)
        '''
        return output


class Judge(MLLM):
    def __init__(self, model, processor, model_family, inference_type):
        super().__init__(model, processor, model_family, inference_type)
        self.system_prompt = ("You will be given an image and a question, corresponding answer, and rationale of that answer (qar)."
                "The qar is synthetically generated. Please serve as an unbiased and fair judge to evaluate the quality of question, answer, and rationale." 
                "Score the response out of 100 and then think and explain in your own words the reasoning for the score with specific details."
                "Use a list-like structure as output: [score, feedback inside double-quotes]. Nothing else.")

    # base-64 encoded image
    def evaluate(self, qars, image):
        if self.model_family == "phi_3_vision":
            for i, qar in enumerate(qars):
                output = self.generate_using_phi3(image, use_evol_prompt=False, questions=str(qar), evolvable_questions=[])
                output = output.strip()
                score, feedback = ast.literal_eval(output)
                qars[i]["score"], qars[i]["feedback"] = score, feedback
        return qars


class BackwardReasoner(MLLM):
    def __init__(self, model, processor, model_family, inference_type):
        super().__init__(model, processor, model_family, inference_type)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.system_prompt = ("You are a helpful assistant designed to infer questions based on visual and textual inputs. "
            "You will be provided with an image, as well as one (answer, rationale) pair. Based on these inputs, your task is to infer "
            "the most appropriate questions that could have led to the given answers and rationales.\n\n"
            "Here is how you will approach the task:\n"
            "1. Analyze the image and understand its context.\n"
            "2. Review the (answer, rationale) pair in connection with the image.\n"
            "3. Infer the possible questions that would logically result in the given answer and rationale within the context of the image.\n\n"
            
            "Rules:\n"
            "1. Ensure that the inferred questions are both relevant to the image and aligned with the (answer, rationale) pair.\n"
            "2. Keep the questions concise and clear.\n"
            "Provide a list of exactly 3 most likely questions for the (answer, rationale) pair.\n\n"
            "Do not hallucinate.\n"
            "Use a list-like structure as output: \"[\"inferred question 1 text\", \"inferred question 2 text\", \"inferred question 3 text\"]\" .Nothing else.")

    def infer_questions(self, image, ar_pairs):
        inferred_questions = []
        for (answer, rationale) in ar_pairs:
            # TODO: Debug
            if self.model_family == "llama_32":
                output = self.generate_using_llama32(image, use_evol_prompt=False, questions=f'[Answer: {answer}, Rationale: {rationale}]', evolvable_questions=[], criteria=CRITERIA, few_shot_questions=FEW_SHOT_QUESTIONS, max_new_tokens=300)
                output = ast.literal_eval(output)
                inferred_questions.append(output)
            else:
                output = self.generate_using_phi3(image, use_evol_prompt=False, questions=f'[Answer: {answer}, Rationale: {rationale}]', evolvable_questions=[])
                output = ast.literal_eval(output)
                inferred_questions.append(output)
        return inferred_questions

    def get_most_similar_question_score(self, question, inferred_questions):
        ques_embed = self.embedding_model.encode(question, convert_to_tensor=True)
        infer_embeds = self.embedding_model.encode(inferred_questions, convert_to_tensor=True)

        # Range: [-1, 1]
        similarity_scores = util.pytorch_cos_sim(ques_embed, infer_embeds)

        # Range: [0, 1]
        similarity_scores = (similarity_scores + 1) / 2
        
        return round(torch.max(similarity_scores).item(), 5)

    # base-64 encoded image
    def verify_inference(self, qars, image):
        questions = [qar["question"] for qar in qars]
        ar_pairs = [(qar["answer"], qar["rationale"]) for qar in qars]
        inferred_questions = self.infer_questions(image, ar_pairs)
        #print(inferred_questions)
        #inferred_questions = [["What is the primary characteristic of a rabbit's ears?", "What is a distinctive feature of a rabbit's eyes?", "What is a notable feature of a rabbit's tail?"], ["What is the rabbit's name?", 'What is the location of the scene?', "What is the rabbit's occupation?"], ['What type of clothing does the rabbit in the image wear?', "What is the color of the rabbit's coat?", "What is the rabbit wearing in the image?"]]
        for i, question in enumerate(questions):
            most_similar_question_score = self.get_most_similar_question_score(question, inferred_questions[i])
            qars[i]["br_score"] = most_similar_question_score * qars[i]["score"]
        return qars

class FinalJudge:
    def __init__(self, model_name, model, conv_template, processor, tokenizer=None, max_length=None):
        print("I'm here")
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = "cuda"
        self.device_map = "auto"
        # "llava_v1" for prometheus vision
        #self.conv_template = "qwen_1_5"
        self.conv_template = conv_template
        self.system_prompt = """You will be given an image and a question related to the image. 

            Evaluate the quality of the question based on the following **criteria** and assign scores for each criterion on a scale of 0-20:
            1. Commonsense knowledge about human social behavior.
            2. Knowledge of the physical world.
            3. Visual understanding.
            4. Reasoning capabilities.
            5. Complexity requiring in-depth reasoning.

            ### Instructions:
            1. For each criterion, assign a score (integer value) between 1 and 5, based on how well the question satisfies the criterion.
            2. Briefly justify the individual criterion scores in one or two sentences.
            4. Identify criteria where the question scored low or failed, with a short explanation.
            5. Suggest a method to evolve (improve) the question to better meet the criteria. Ensure that evolving the question to improve certain aspects does **not compromise or lower the quality of other aspects**.

            ### Important Note:
            Instead of an image description, an **actual image** will be provided for evaluation. Use the image content to assess the question and assign scores. DO NOT hallucinate or infer details that cannot be directly observed or logically deduced from the image.
            
            ### Output Format:
            Provide the evaluation in the following structured format. Do NOT use strict JSON, but keep the structure clear and detectable:

            - **Scores**:
              - Commonsense: <score>
              - Physical World: <score>
              - Visual Understanding: <score>
              - Reasoning: <score>
              - Complexity: <score>
            - **Justification**: <brief justification of the individual scores>
            - **Failures**: <brief explanation of criteria not fully met>
            - **Evolution Method**: <method to evolve (improve) the question without lowering the quality of other aspects>

            
            Neither favor nor punish due to your internal bias. DO FAIR JUDGEMENT."""

    
    # base-64 encoded image
    def evaluate(self, image, qars):
        # Use LLaVA-Critic
        if self.model_name == "llava_critic":
            image_tensor = process_images([image], self.processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            #print(image_tensor[0].dtype)
            
            kk = 0
            for i, qar in enumerate(qars):
                query = f'{DEFAULT_IMAGE_TOKEN}\nQuestion (to be judged): {qar["question"]}'
                conv = copy.deepcopy(conv_templates[self.conv_template])
                conv.append_message(conv.roles[1], self.system_prompt)
                conv.append_message(conv.roles[0], query)
                input_text = conv.get_prompt()

                input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
                image_sizes = [image.size]

                #print(input_ids)
                
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    temperature=0.1,
                    max_new_tokens=1000,
                )
                
                judgement_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                qars[i]["judgement_details"] = judgement_text
        elif self.model_name == "prometheus_vision":
            image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            for i, qar in qars:
                query = f'Question (to be judged): {qar["question"]}'
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[1], self.system_prompt)
                conv.append_message(conv.roles[0], query)
                conv.append_message(conv.roles[1], None)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                input_text = conv.get_prompt()
                
                input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=True,
                        temperature=0.1,
                        num_beams=1,
                        max_new_tokens=1000,
                    )

                judgement_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                judgement_text = judgement_text.strip()
                if judgement_text.endswith(stop_str):
                    judgement_text = outputs[:-len(stop_str)]
                judgement_text = judgement_text.strip()
                qars[i]["judgement_details"] = judgement_text
        # Use LLaVA-Next
        elif self.model_name == "llava_next":
            for i, qar in enumerate(qars):
                query = f'Question (to be judged): {qar["question"]}'
                
                messages = [
                    { 
                        "role": "assistant", 
                        "content": [
                            {"type": "text", "text": self.system_prompt}
                        ] 
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": query}
                        ]
                    },
                ]

                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(
                    image,
                    input_text,
                    return_tensors="pt"
                ).to(self.model.device)

                output = self.model.generate(
                    **inputs,
                    temperature=1.0, 
                    max_new_tokens=1000
                )

                judgement_text = self.processor.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                qars[i]["judgement_details"] = judgement_text
        
            #judgement_text = self.parse_judgement(judgement_text[0])
            #qars[i]['evaluation'] = judgement_text
        return qars

    def parse_judgement(self, judgement):
        json_structure = re.search(r'```json\n({.*?})\n```', judgement, re.DOTALL)
        if json_structure:
            json_object = json_structure.group(1)
            parsed_judgement = json.loads(json_object)
        return parsed_judgement
            