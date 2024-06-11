from .base import *


class LlamacppHelper(BaseHelper):

    def __init__(self, sync, model=None, path_to_model=None):
        assert ((model!=None) or path_to_model!=None), f"must provide either loaded model or path to model files, got: model={model}, path_to_model={path_to_model}"
        """
        if model provided starts in already loaded mode,
        otherwise uses path_to_model to load the model
        """
        
        self.sync = sync
        self.model = model
        self.path_to_model = path_to_model
        
    def encode(self, inputs: Union[str, List[str]], encode_special_tokens=True) -> EncodeOutputDict:
        """
        encoded strings to tokens
        
        input:  string or list of strings
        output: EncodeOutputDict containing 
                2D tensor of (batch, tokens) and mask for it
        """

        print(f"encoder input: {inputs}")
        if isinstance(inputs, str):
            inputs = [inputs]

        encoded_merker = []
        for entry in inputs:
            entry_encoded = self.model.tokenize(entry.encode('UTF-8'), special=True, add_bos = False)
            encoded_merker.append(entry_encoded)

        print("encoded_merker:", encoded_merker)        
        
        return encoded_merker
        

    def decode(self, inputs: torch.Tensor, skip_special_tokens=True, logits_mode=False) -> Union[List[str], List[List[str]]]:
        """
        decoded tokens to strings

        input: 1D or 2D tensor containing tokens
        output: 2D list of strings
        """

        print(f"decoder input: {inputs}")
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        decoded_merker = []
        for entry in inputs:
            if logits_mode:
                decoded_merker.append([])
                for subentry in entry.tolist():
                    decoded_merker[-1].append(self.model.detokenize([subentry]).decode('UTF-8'))
            else:
                entry_decoded = self.model.detokenize(entry).decode('UTF-8')
                decoded_merker.append(entry_decoded)

        if logits_mode:
            decoded_merker = decoded_merker[0]

        print("decoded_merker:", decoded_merker)
        

        return decoded_merker
        
    
    def generate(self, inputs: torch.Tensor, **kwargs) -> GenerateOutputDict:
        """
        generates text using the LLM model

        input:  2D tensor containing tokens, kwargs 
                for generation settings
        output: GenerateOutputDict containing 
                decoded output string, output shape 
                and top_logits
        """

        # make sure inputs tensor is 2D
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        input_length = inputs.shape[-1]
            
        # need to process batches sequentially
        out_merker = []
        # print("inputs:", inputs)
        for entry in inputs:
            entry_out = self.model(entry.tolist(), **kwargs)
            out_merker.append(entry_out)

        # print("out_merker:")
        # show_dict_compact(out_merker)
        # print("text_offset:", out_merker[0]['choices'][0]['logprobs']['text_offset'])
        # print(out_merker[0]['choices'][0]['logprobs']['top_logprobs'][0])
        # print("top_logprobs:", [f"{key}: {out_merker[0]['choices'][0]['logprobs']['top_logprobs'][0][key]}" for index, key in enumerate(out_merker[0]['choices'][0]['logprobs']['top_logprobs'][0]) if index < 10])

        decoded = []
        for entry in out_merker:
            decoded.append(entry['choices'][0]['text'])

        output_shape = [len(out_merker), len(out_merker[0]['choices'][0]['logprobs']['top_logprobs'])]

        top_logits = torch.tensor([[[[self.encode(key)[0][0], val] for key, val in dict(list(top_logits.items())[:self.sync.dhold.inputs['max_num_beams']]).items()] for top_logits in out['choices'][0]['logprobs']['top_logprobs']] for out in out_merker])

        
        print("top_logits:", top_logits)
        

        
        output = GenerateOutputDict(
            decoded_output = decoded,
            output_shape = output_shape,
            logits = None,
            top_logits = top_logits
        )

        print("output GenerateOutputDict:")
        show_dict_compact(output)
        
        return output