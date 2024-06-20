from .imports import *

from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline


class ModelHolder():
    def __init__(self):
        self.current_model = ""
        self.current_dtype = ""

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

        self.current_model = model_name
        self.current_dtype = dtype
        pretrained = sync.config['models'][model_name]['path']
        backend = sync.config['models'][model_name]['backend']
        base_model = sync.config['models'][model_name]['base_model']
        
        use_processor = False

        if base_model == "SD-1.5":
            self.model = StableDiffusionPipeline.from_single_file(pretrained, torch_dtype=torch.float16).to("cuda")
        elif base_model == "SD-XL":
            self.model = StableDiffusionXLPipeline.from_single_file(pretrained, torch_dtype=torch.float16).to("cuda")
        else:
            self.model = AutoPipelineForText2Image.from_pretrained(pretrained, torch_dtype=torch.float16).to("cuda")
