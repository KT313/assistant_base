from .imports import *

class ModelHolder():
    def __init__(self):
        self.current_model = ""
        self.current_dtype = ""

    def load_model(self, sync, model_name, dtype):
        try:
            del self.model
            gc.collect()
        except:
            pass
            
        if model_name == "llama3-llava-next-8b":
            pretrained = sync.config['models'][model_name]['path']
            model_name_for_loading = "llava_llama3"
            tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name_for_loading, device_map=sync.config['torch_device_map'])
            
            model.eval()
            model.tie_weights()
            
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
            self.max_length = max_length
    
        if model_name == "paligemma-3b-mix-448":
            pretrained = sync.config['models'][model_name]['path']
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=sync.config['torch_device_map'],
                revision="bfloat16",
            ).eval()
            processor = AutoProcessor.from_pretrained(pretrained)
    
            self.processor = processor
            self.model = model
    
        if model_name == "Meta-Llama-3-70B-Instruct-IQ2_S":
            pretrained = sync.config['models'][model_name]['path']
            model = Llama(
                model_path=pretrained,
                n_gpu_layers=-1, # Uncomment to use GPU acceleration
                # seed=1337, # Uncomment to set a specific seed
                n_ctx=1024, # Uncomment to increase the context window
                flash_attn=True,
            )
            self.model = model
    
        if model_name == "Meta-Llama-3-70B-Instruct-IQ1_M":
            pretrained = sync.config['models'][model_name]['path']
            model = Llama(
                model_path=pretrained,
                n_gpu_layers=-1, # Uncomment to use GPU acceleration
                # seed=1337, # Uncomment to set a specific seed
                n_ctx=1024, # Uncomment to increase the context window
                flash_attn=True,
            )
            self.model = model
    
        if model_name == "Hermes-2-Theta-Llama-3-8B":
            pretrained = sync.config['models'][model_name]['path']
            tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=False, padding_side='left')
    
            if dtype == "float16":
                model = LlamaForCausalLM.from_pretrained(
                    pretrained,
                    torch_dtype=torch.float16,
                    device_map=sync.config['torch_device_map'],
                    # quantization_config=bnb_config,
                    use_flash_attention_2=True
                )
            elif dtype == "bfloat16":
                model = LlamaForCausalLM.from_pretrained(
                    pretrained,
                    torch_dtype=torch.bfloat16,
                    device_map=sync.config['torch_device_map'],
                    # quantization_config=bnb_config,
                    use_flash_attention_2=True
                )
            else:
                if dtype == "8bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                elif dtype == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                
                model = LlamaForCausalLM.from_pretrained(
                    pretrained,
                    # torch_dtype=torch.float16,
                    device_map=sync.config['torch_device_map'],
                    quantization_config=bnb_config,
                    use_flash_attention_2=True
                )
            
            self.tokenizer = tokenizer
            self.model = model

        if model_name == "phi-3-vision-128k-instruct":
            pretrained = sync.config['models'][model_name]['path']
            self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
            self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True) 
            
        self.current_model = model_name
        self.current_dtype = dtype