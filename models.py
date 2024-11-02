import ast
import pprint
import copy

import torch
from transformers import GenerationConfig

from sentence_transformers import SentenceTransformer, util

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

class MLLM:
    def __init__(self, model, processor, model_family, inference_type):
        self.model = model
        self.processor = processor
        self.model_family = model_family 
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

    def generate_using_llama32(self, image, query, questions=None):
        if questions is None:
            messages = [
                {
                    "role": "assistant",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                },
            ]

        else:
            messages = [
                {
                    "role": "assistant",
                    "content": self.system_prompt
                },
                {
                    "role": "assistant",
                    "content": questions
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                },
            ]

        '''
        {
            "role": "assistant",
            "content": "[{'question': 'What is the animal depicted in the image?', 'answer': 'A rabbit'}, {'question': 'Where is the rabbit standing?', 'answer': 'On a dirt path'}, {'question': 'What is the rabbit wearing?', 'answer': 'A blue coat'}]"
        },
        {
            "role": "user",
            "content": "Can you generate corresponding rationales for each question-answer pair, that justify briefly why the answer is correct based on the image?"
        }
        '''

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

    def generate_using_phi3(self, image, query, questions=None):
        placeholder = ""
        placeholder += f"<|image_1|>\n"

        messages = [
            {"role": "assistant", "content": self.system_prompt},
            {"role": "user", "content": placeholder},
            {"role": "user", "content": query}
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
            
    # base-64 encoded image
    def generate(self, image, query, questions=None):
        output = None
        if self.model_family == "llama_32":
            output = self.generate_using_llama32(image, query, questions)
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
                output = self.generate_using_phi3(image, str(qar))
                output = output.strip()
                score, feedback = ast.literal_eval(output)
                qars[i]["score"], qars[i]["feedback"] = score, feedback
        return qars


class BackwardReasoner(MLLM):
    def __init__(self, model, processor, model_family, inference_type):
        super().__init__(model, processor, model_family, inference_type)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.system_prompt = ("You are a helpful assistant designed to infer questions based on visual and textual inputs. "
            "You will be provided with an image, as well as  one (answer, rationale) pair. Based on these inputs, your task is to infer "
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
                output = self.generate_using_llama32(image, f'[{answer}, {rationale}]')
                output = ast.literal_eval(output)
                inferred_questions.append(output)
            else:
                pass
                #output = self.generate_using_phi3(image, f'[{answer}, {rationale}]')
            #print(output)
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
        #inferred_questions = self.infer_questions(image, ar_pairs)
        #print(inferred_questions)
        inferred_questions = [["What is the primary characteristic of a rabbit's ears?", "What is a distinctive feature of a rabbit's eyes?", "What is a notable feature of a rabbit's tail?"], ["What is the rabbit's name?", 'What is the location of the scene?', "What is the rabbit's occupation?"], ['What type of clothing does the rabbit in the image wear?', "What is the color of the rabbit's coat?", "What is the rabbit wearing in the image?"]]
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
            "- Now propose a question evolving method which would address the shortcoming and overall improve the question.\n\n"
            
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
    
        for qar in qars:
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

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            dsp = {
                'question': qar["question"],
                'eval': text_outputs[0]
            }

            pprint.pprint(dsp)
            print("x"*30)
            