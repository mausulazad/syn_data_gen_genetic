import ast

from transformers import GenerationConfig


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

    def generate_using_llama32(self, image, query, questions):
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
                "Use a list-like structure as output: [<score>, <feedback>]. Nothing else.")

    # base-64 encoded image
    def judge_qars(self, qars, image):
        if self.model_family == "phi_3_vision":
            for qar in qars:
                output = self.generate_using_phi3(image, str(qar))
                output = output.strip()
                score, feedback = ast.literal_eval(output)
                qar["score"], qar["feedback"] = score, feedback


class BackwardReasoner(MLLM):
    def __init__(self, model, processor, model_family, inference_type):
        super().__init__(model, processor, model_family, inference_type)

    # base-64 encoded image
    def verify_inference(self, answer, rationale, image):
        pass