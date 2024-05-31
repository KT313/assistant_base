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

        # TODO: currently dtype only for hermes model, also make for others
        if dtype == "float16": torch_dtype = torch.float16
        elif dtype == "bfloat16": torch_dtype = torch.bfloat16
        else: torch_dtype = None
        
        if dtype == "8bit": bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif dtype == "4bit": bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        else: bnb_config = None

        self.current_model = model_name
        self.current_dtype = dtype
        pretrained = sync.config['models'][model_name]['path']

        if model_name in ["llama3-llava-next-8b"]:
            self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(pretrained, None, "llava_llama3", device_map=sync.config['torch_device_map'])
            self.model.eval()
            self.model.tie_weights()
        if model_name in ["Hermes-2-Theta-Llama-3-8B"]:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=False, padding_side='left')
            self.model = LlamaForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map=sync.config['torch_device_map'], quantization_config=bnb_config, use_flash_attention_2=True)
        if model_name in ["phi-3-vision-128k-instruct"]:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="cuda", trust_remote_code=True, torch_dtype="auto").eval()
            self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True) 
        if model_name in ["Meta-Llama-3-70B-Instruct-IQ2_S", "Meta-Llama-3-70B-Instruct-IQ1_M"]:
            self.model = Llama(model_path=pretrained, n_gpu_layers=-1, n_ctx=1024, logits_all=True, flash_attn=True)
            
        self.backend = sync.config['models'][model_name]['backend']
        self.template = sync.config['models'][model_name]['template']
        self.image_capable = sync.config['models'][model_name]['image-capable']



        return None