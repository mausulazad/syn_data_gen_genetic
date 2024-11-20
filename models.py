import re
import json

import ast
import pprint
import copy

import torch
from transformers import GenerationConfig

from sentence_transformers import SentenceTransformer, util

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from transformers import pipeline

class MLLM:
    def __init__(self, model, processor, model_family, inference_type):
        self.model = model
        self.processor = processor
        self.model_family = model_family
        self.inference_type = inference_type 
        self.system_prompt = self.get_system_prompt(inference_type)
    
    def get_system_prompt(self, inference_type):
        prompt = None
        if inference_type == "generate":
            prompt = "You are an helpful assistant. You help people in generating synthetic questions/answers/rationales (as instructed by the user) related to images."
        # LATER: Modify system prompt
        elif inference_type == "judge":
            prompt = "You are an helpful assistant"
        # TODO
        elif inference_type == "backward_reasoning":
            prompt = "You are an helpful assistant. Given an answer and image, you try to infer the corresponding question."
        return prompt

    def generate_using_llama32(self, image, use_evol_prompt, questions, evolvable_questions):
        if self.inference_type == "generate":
            # Generating questions
            if questions is None:
                # Evolving generated questions
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
                # 1st time generating questions
                else:
                    query = "Generate 3 non-trivial, diverse questions (add '?' after each question) based on the image without hallucinating that can be answered using one to at max. five words. Each question can not have sub-questions. Return the questions in a list (comma separated) like this: [<question 1>, <question 2>, <question 3>]. Return only the list of questions, nothing else."
                
                messages = [
                    { "role": "assistant", "content": self.system_prompt },
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
                query = 'You will be given an image and questions that are based on the image. For each question generate correct answer and corresponding brief rationale (rationale should justify briefly why the answer is correct) without hallucinating. Keep the answers precise and short (no over explanation). Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. Return only the list of JSON objects, nothing else.'
                messages = [
                    { "role": "assistant", "content": self.system_prompt },
                    { "role": "assistant", "content": questions },
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

        output = self.model.generate(**inputs, max_new_tokens=300)
        output = self.processor.decode(output[0][inputs.input_ids.shape[-1]:])
        
        # llama-32 specific
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


    def generate_using_llava_next(self, image, use_evol_prompt, questions, evolvable_questions):
        # Generating questions
        if questions is None:
            # Evolving generated questions
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
            # 1st time generating questions
            else:
                query = "Generate 3 non-trivial, diverse questions (add '?' after each question) based on the image without hallucinating that can be answered using one to at max. five words. DO NOT ANSWER, JUST GENERATE QUESTIONS. Each question can not have sub-questions. Return all questions inside a list (comma separated) like this: [<question 1>, <question 2>, <question 3>]. Return only the list of questions (generate 3 questions, NOT LESS THAN THAT. AND MUST RETURN THEM INSIDE A LIST.), nothing else."
                
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
        # Generating answers and rationales
        else:
            query = 'Given an image and questions based on the image, generate a correct answer and a corresponding brief but insightful rationale for each question. The rationale should justify why the answer is correct by referencing specific details in the image or using logical inference when appropriate. Avoid vague statements; each rationale should clarify how the visible elements or context in the image supports the answer. Ensure answers are precise and concise (1-2 words if possible), and the rationale directly connects to the answer without over-explanation. Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. YOU MUST RETURN ALL NON-DUPLICATE JSON OBJECTS INSIDE A LIST.Return only the list of JSON objects, nothing else.'
            
            messages = [
                { 
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": self.system_prompt}
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
        output = self.model.generate(**inputs, max_new_tokens=300)
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

    def generate_using_molmo(self, image, use_evol_prompt, questions, evolvable_questions):
        if questions is None:
            # TODO: For tries > 0
            if use_evol_prompt:
                pass
            else:
                query = "Can you generate 3 non-trivial, diverse questions based on the image without hallucinating that can be answered using one to at max. five words?  Each question can not have sub-questions. Return the questions in a list (MUST BE comma separated, YOU FORGET THIS, DON'T) like this: [<question 1>, <question 2>, <question 3>]. Return only the list of questions, nothing else."
        else:
            query = 'You will be given an image and questions that are based on the image. For each question generate correct answer and corresponding brief rationale (rationale should justify briefly why the answer is correct) without hallucinating. Keep the answers precise and short (no over explanation). Return the question, answer, rationale triplets in a list following this structure: [{"question": <given question>, "answer": <corresponding correct answer>, "rationale": <corresponding rationale>}]. Return only the list of JSON objects, nothing else. REMEMBER, YOU HAVE TO DO THIS FOR EACH GIVEN QUESTION, NOT JUST FIRST ONE.'
        
        prompt = None
        if questions is None:
            prompt = f'{self.system_prompt}\n\n{query}'
        else:
            prompt = f'{self.system_prompt}\n\n{query}\n\n{questions}'

        inputs = self.processor.process(
            images=[image],
            text=prompt
        )

        inputs["images"] = inputs["images"].to(torch.bfloat16)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=300, temperature=0.3, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        generated_tokens = output[0,inputs['input_ids'].size(1):]
        output = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output

    # base-64 encoded image
    def generate(self, image, use_evol_prompt=False, questions=None, evolvable_questions=[]):
        output = None
        if self.model_family == "llama_32":
            output = self.generate_using_llama32(image, use_evol_prompt, questions, evolvable_questions)
        elif self.model_family == "molmo":
            output = self.generate_using_molmo(image, use_evol_prompt, questions, evolvable_questions)
        elif self.model_family == "llava":
            output = self.generate_using_llava(image, use_evol_prompt, questions, evolvable_questions)
        elif self.model_family == "llava_next":
            output = self.generate_using_llava_next(image, use_evol_prompt, questions, evolvable_questions)
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
            if self.model_family == "llama_32":
                output = self.generate_using_llama32(image, use_evol_prompt=False, questions=f'[Answer: {answer}, Rationale: {rationale}]', evolvable_questions=[])
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
    def __init__(self, model, processor, tokenizer, max_length):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = "cuda"
        self.device_map = "auto"
        self.conv_template = "qwen_1_5"
        self.system_prompt = ("You will be given a question related to the image.\n"
            "Evaluate the quality of the question for challenging advanced reasoning systems like ChatGPT or GPT-4 on following criteria:\n"
            "- Add 20 points if the question requires commonsense knowledge about human social behavior to answer, otherwise add 0.\n"
            "- Add 20 points if the question requires knowledge of the physical world to answer, otherwise add 0.\n"
            "- Add 20 point if visual understanding is necessary to answer the question, otherwise add 0.\n"
            "- Add 20 point if the question challenges the system's reasoning capabilities, otherwise add 0.\n"
            "- Add 20 point if the question is sufficiently complex to require in-depth reasoning, otherwise add 0.\n\n"
            "Sum up the score (do not hallucinate, do proper arithmetic operation, can be at max 100). After scoring out of 100, examine the question and identify the cases it failed to meet:\n"
            "- First, Justify your total score, up to 100 words.\n"
            "- Next, briefly discuss the criterion the question failed to meet in under 100 words.\n"
            "- Now propose a question evolving method which would address the shortcoming and overall improve the question, WITHOUT directly providing example questions.\n\n"
            
            "DO NOT, I REPEAT, DO NOT ANSWER THE QUESTION. YOU MUST JUDGE IT, BASED ON GIVEN CRITERION."
            "Finally, return the score, justification, failures, evolution method as a JSON STRUCTURE. I NEED TO PARSE IT LATER," 
            "SO DONT ADD ANYTHING ELSE AFTER OR BEFORE THE JSON OBJECT, OTHERWISE IT WILL BREAK MY CODE AND I WILL BE VERY SAD.\n"
            "- Please do NOT add new lines or tabs in the JSON.\n"
            "- Always return a valid parse-able JSON STRUCTURE. YOU ALWAYS FORGET THAT. THIS WILL CAUSE A FIRE ON A PRODUCTION SERVER BEING USED BY MILLIONS.\n"
            "The JSON structure will have following key-value items:\n"
            "1. score: <Total score out of 100>\n"
            "2. justification: <Justification of given score>\n"
            "3. failures: <Brief desciption of given criterion that are not present in question>\n"
            "4. evolution_method: <A question evolving method that can be used to generate an evolved question that meets all/most criterion>\n")

    
    # base-64 encoded image
    def evaluate(self, qars, image):
        image_tensor = process_images([image], self.processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
    
        for i, qar in enumerate(qars):
            question = f'{DEFAULT_IMAGE_TOKEN}\nThis is the question to judge: {qar["question"]}'
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[1], self.system_prompt)
            conv.append_message(conv.roles[0], question)
            judge_prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(judge_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            image_sizes = [image.size]

            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.3,
                max_new_tokens=4096,
            )

            judgement_text = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            judgement_text = self.parse_judgement(judgement_text[0])
            qars[i]['evaluation'] = judgement_text

        return qars

    def parse_judgement(self, judgement):
        json_structure = re.search(r'```json\n({.*?})\n```', judgement, re.DOTALL)
        if json_structure:
            json_object = json_structure.group(1)
            parsed_judgement = json.loads(json_object)
        return parsed_judgement
            