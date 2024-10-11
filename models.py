class MLLM:
    def __init__(self, model, processor, inference_type):
        self.model = model
        self.processor = processor
        self.system_prompt = self.get_system_prompt(inference_type)
    
    def get_system_prompt(self, inference_type):
        prompt = None
        if inference_type == "generate":
            prompt = "You are an helpful assistant. You help people in answering questions related to images."
        elif inference_type == "judge":
            prompt = "You are an helpful assistant. You fairly judge peoples' answers, given the question and corresponding image."
        elif inference_type == "backward_reasoning":
            prompt = "You are an helpful assistant. Given an answer and image, you try to infer the corresponding question."
        return prompt
            
    # base-64 encoded image
    def generate(self, image, query):
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
            {
                "role": "assistant",
                "content": "[{'question': 'What is the animal depicted in the image?', 'answer': 'A rabbit'}, {'question': 'Where is the rabbit standing?', 'answer': 'On a dirt path'}, {'question': 'What is the rabbit wearing?', 'answer': 'A blue coat'}]"
            },
            {
                "role": "user",
                "content": "Can you generate corresponding rationales for each question-answer pair, that justify briefly why the answer is correct based on the image?"
            }
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=300)
        print(self.processor.decode(output[0]))



class Judge(MLLM):
    def __init__(self, model, processor, inference_type):
        super().__init__(model, processor, inference_type)

    # base-64 encoded image
    def judge_qars(self, qars, image):
        pass


class BackwardReasoner(MLLM):
    def __init__(self, model, processor, inference_type):
        super().__init__(model, processor, inference_type)

    # base-64 encoded image
    def verify_inference(self, answer, rationale, image):
        pass