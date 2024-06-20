from .base import *
from llava.constants import IMAGE_TOKEN_INDEX


class TransformersHelper(BaseHelper):

    def __init__(self, sync, model=None, tokenizer=None, image_processor=None, path_to_model=None):
        assert ((model!=None and tokenizer!=None) or path_to_model!=None), f"must provide either loaded model and tokenizer or path to model files, got: model={model}, tokenizer={tokenizer}, path_to_model={path_to_model}"
        """
        if model and tokenizer provided starts in already loaded mode,
        otherwise uses path_to_model to load the model and tokenizer
        """
        
        self.sync = sync
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.path_to_model = path_to_model

        self.stop_token = torch.tensor([self.tokenizer.eos_token_id, 128001, 128009], device=self.sync.config['torch_device'])
        
    def encode(self, inputs: Union[str, List[str]], images=None, encode_special_tokens=True) -> EncodeOutputDict:
        """
        encoded strings to tokens
        
        input:  string or list of strings
        output: EncodeOutputDict containing 
                2D tensor of (batch, tokens) and mask for it
        """
        
        if images != None and len(images) >= 1:
            if self.sync.dhold.inputs['debugmode']: print("found images to encode")
            if self.sync.mhold.current_model == "llama3-llava-next-8b":
                from transformers import AutoProcessor
                from llava.mm_utils import tokenizer_image_token

                processor = AutoProcessor.from_pretrained(self.sync.config['models'][self.sync.mhold.current_model]['path'], trust_remote_code=True)
                tokenizer_output = processor(inputs, images, return_tensors="pt").to(self.sync.config['torch_device'])

                input_ids = tokenizer_image_token(inputs, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.sync.config['torch_device'])
                

                output = EncodeOutputDict(
                    ids = input_ids,
                    mask = tokenizer_output.attention_mask,
                    images = [tokenizer_output.pixel_values.to(torch.float16)],
                    image_sizes = [image.size for image in images]
                )
                if self.sync.dhold.inputs['debugmode']: print("encoded images for llama3-llava")
                
            elif self.sync.mhold.current_model == "phi-3-vision-128k-instruct":
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(self.sync.config['models'][self.sync.mhold.current_model]['path'], trust_remote_code=True)
                tokenizer_output = processor(inputs, images, return_tensors="pt").to(self.sync.config['torch_device'])

                output = EncodeOutputDict(
                    ids = tokenizer_output.input_ids,
                    mask = torch.ones_like(tokenizer_output.input_ids, device=self.sync.config['torch_device']),
                    pixel_values = tokenizer_output.pixel_values if "pixel_values" in tokenizer_output else None,
                    image_sizes = tokenizer_output.image_sizes if "image_sizes" in tokenizer_output else None
                )
                if self.sync.dhold.inputs['debugmode']: print("encoded images for phi-3-vision")
                
            else:
                # not vision capable model, ignore images
                encoded_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.sync.config['torch_device'])
        
                output = EncodeOutputDict(
                    ids = encoded_ids,
                    mask = torch.ones_like(encoded_ids, device=self.sync.config['torch_device']),
                )

        else:
        
            encoded_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.sync.config['torch_device'])
        
            output = EncodeOutputDict(
                ids = encoded_ids,
                mask = torch.ones_like(encoded_ids, device=self.sync.config['torch_device']),
            )
            
        return output
        

    def decode(self, inputs: torch.Tensor, skip_special_tokens=True, logits_mode=False) -> Union[List[str], List[List[str]]]:
        """
        decoded tokens to strings

        input: 1D or 2D tensor containing tokens
        output: 2D list of strings
        """
        
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if logits_mode:
            decoded = [self.tokenizer.batch_decode(entry, skip_special_tokens=skip_special_tokens) for entry in inputs]
        else:
            decoded = [self.tokenizer.batch_decode(inputs, skip_special_tokens=skip_special_tokens)]
        
        if logits_mode:
            decoded = decoded[0]

        return decoded
        
    
    def generate(self, inputs: torch.Tensor, **kwargs) -> GenerateOutputDict:
        """
        generates text using the LLM model

        input:  2D tensor containing tokens, kwargs 
                for generation settings
        output: GenerateOutputDict containing 
                decoded output string, output shape 
                and top_logits
        """

        assert (inputs.ndim == 2 or inputs.ndim == 1), f"inputs need to be 1D or 2D tensor (batch, tokens), got: {inputs.shape}"

        # make sure inputs tensor is 2D
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        
        out = self.model.generate(inputs, **kwargs)

        input_sequence_length = inputs.shape[-1]
        # llama-llava does for some reason not return the output with input and i did not find a way to check the amount of actually new generated tokens, so for now this should at least work. TODO: find cleaner solution than only for this specific model
        if self.sync.mhold.current_model in ["llama3-llava-next-8b"]:
            input_sequence_length = 0
        
        decoded = self.decode(out.sequences[:, input_sequence_length:], skip_special_tokens=True)

        output_shape = out.sequences.shape
        merker_scores = []
        if out.scores != None:
            for i in range(len(out.scores)):
                merker_scores.append(out.scores[i])
            scores = torch.stack(merker_scores)
            scores = scores.permute(1, 0, 2) # so the order is batch, token, scores

            top_logits = find_top_indexes(self.sync, scores)
        else:
            logits = None
            top_logits = None

        
        output = GenerateOutputDict(
            decoded_output = decoded,
            output_shape = output_shape,
            logits = None,
            top_logits = top_logits
        )
        
        return output