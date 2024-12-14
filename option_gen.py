cache_dir= '/scratch/mi8uu/cache'
os.environ['TRANSFORMERS_CACHE']=cache_dir
os.environ['HF_HOME']= cache_dir

def generate_options():
    vlm_id = "microsoft/Phi-3.5-vision-instruct" 
    vlm = AutoModelForCausalLM.from_pretrained(
        vlm_id, 
        #device_map="cuda",
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2',
        cache_dir=cache_dir
    )

    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        num_crops=16,
        cache_dir=cache_dir
    )
    vlm = None
    processor = None
    system_prompt

if __name__ == "__main__":
    generate_options()