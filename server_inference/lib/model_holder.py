from .imports import *

class ModelHolder():
    def __init__(self):
        self.current_model = ""
        self.current_dtype = ""
        self.preprocessor_type = ""

    def load_model(self, sync, model_name, dtype):
        try:
            del self.model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
            
        if model_name == "llama3-llava-next-8b":
            pretrained = sync.config['models'][model_name]['path']
            model_name_for_loading = "llava_llama3"
            tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name_for_loading, device_map=sync.config['torch_device_map'])
            
            model.eval()
            model.tie_weights()
            
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
            self.preprocessor_type = "tokenizer+image_processor"
    
        if model_name == "Meta-Llama-3-70B-Instruct-IQ2_S" or model_name == "Meta-Llama-3-70B-Instruct-IQ1_M":
            pretrained = sync.config['models'][model_name]['path']
            model = Llama(
                model_path=pretrained,
                n_gpu_layers=-1,
                n_ctx=1024,
                logits_all=True,
                flash_attn=True,
            )
            self.model = model
            self.preprocessor_type = "llama_cpp"
    
        if model_name == "Hermes-2-Theta-Llama-3-8B":
            pretrained = sync.config['models'][model_name]['path']
            tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=False, padding_side='left')
    
            if dtype == "float16": torch_dtype = torch.float16
            elif dtype == "bfloat16": torch_dtype = torch.bfloat16
            else: torch_dtype = None
            
            if dtype == "8bit": bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif dtype == "4bit": bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            else: bnb_config = None
                
            model = LlamaForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map=sync.config['torch_device_map'], quantization_config=bnb_config, use_flash_attention_2=True)
            
            self.tokenizer = tokenizer
            self.model = model
            self.preprocessor_type = "tokenizer"

        if model_name == "phi-3-vision-128k-instruct":
            pretrained = sync.config['models'][model_name]['path']
            self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="cuda", trust_remote_code=True, torch_dtype="auto").eval()
            self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True) 
            self.preprocessor_type = "processor"
            
        self.current_model = model_name
        self.current_dtype = dtype