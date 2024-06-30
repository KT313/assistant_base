from .imports import *

from .backends.transformers import TransformersHelper
from .backends.llama_cpp import LlamacppHelper
from .backends.exllamav2 import Exllamav2Helper

class ModelHolder():
    def __init__(self):
        self.current_model = ""
        self.current_dtype = ""
        self.preprocessor_type = ""

    def load_model(self, sync, model_name, dtype):
        """
        loads a model using the provided information in sync.dhold.inputs['model'] and stores it and its components (tokenizer and maybe image_processor) in sync.mhold
        """
        
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
        backend = sync.config['models'][model_name]['backend']
        use_processor = False


        if model_name in ["llama3-llava-next-8b"]:
            self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(pretrained, None, "llava_llama3", device_map=sync.config['torch_device_map'])
            self.model.eval()
            self.model.tie_weights()
            use_processor=True

        elif model_name in ["phi-3-vision-128k-instruct"]:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="cuda", trust_remote_code=True, torch_dtype="auto").eval()
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True) 
            
            
        elif backend in ["transformers"]:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=False, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map=sync.config['torch_device_map'], quantization_config=bnb_config, attn_implementation="flash_attention_2")
            
            
        elif backend in ["llama-cpp"]:
            self.model = Llama(model_path=pretrained, n_gpu_layers=-1, n_ctx=1024, verbose=False, logits_all=True, flash_attn=True)


        elif backend in ["exllamav2"]:
            config = ExLlamaV2Config(pretrained)
            self.model = ExLlamaV2(config)
            self.cache = ExLlamaV2Cache(self.model, max_seq_len = 4096, lazy = True)
            self.model.load_autosplit(self.cache, progress = True)
            self.tokenizer = ExLlamaV2Tokenizer(config)

        
        try:
            self.model.eval()
        except:
            print(f"could not send model {model_name} to eval mode")
            
        self.backend = sync.config['models'][model_name]['backend']
        self.template = sync.config['models'][model_name]['template']
        self.image_capable = sync.config['models'][model_name]['image-capable']


        if self.backend == "transformers":
            if use_processor:
                self.helper = TransformersHelper(sync=sync, model=self.model, tokenizer=self.tokenizer, image_processor=self.image_processor)
            else:
                self.helper = TransformersHelper(sync=sync, model=self.model, tokenizer=self.tokenizer)
        elif self.backend == "llama-cpp":
            self.helper = LlamacppHelper(sync=sync, model=self.model)
        elif self.backend == "exllamav2":
            self.helper = Exllamav2Helper(sync=sync, model=self.model, tokenizer=self.tokenizer, cache=self.cache)
