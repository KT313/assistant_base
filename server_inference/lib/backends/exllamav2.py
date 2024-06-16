from .base import *
import random
import copy
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
from exllamav2.generator import ExLlamaV2Sampler


class Exllamav2Helper(BaseHelper):

    def __init__(self, sync, model=None, tokenizer=None, cache=None, path_to_model=None):
        assert ((model!=None and tokenizer!=None and cache!=None) or path_to_model!=None), f"must provide either loaded model and tokenizer and cache or path to model files, got: model={model}, tokenizer={tokenizer}, cache={cache}, path_to_model={path_to_model}"
        """
        if model and tokenizer provided starts in already loaded mode,
        otherwise uses path_to_model to load the model and tokenizer
        """
        
        self.sync = sync
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.path_to_model = path_to_model
        self.generator = ExLlamaV2DynamicGenerator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
        )

        self.stop_token = torch.tensor([self.tokenizer.eos_token_id, self.encode("<|eot_id|>", encode_special_tokens=True)['ids'][0, 0]], device=self.sync.config['torch_device'])


        
        
    def encode(self, inputs: Union[str, List[str]], images=None, encode_special_tokens=True) -> EncodeOutputDict:
        """
        encoded strings to tokens
        
        input:  string or list of strings
        output: EncodeOutputDict containing 
                2D tensor of (batch, tokens) and mask for it
        """

        if isinstance(inputs, str):
            inputs = [inputs]

        tokens, position_offsets = self.tokenizer.encode(
            inputs,
            encode_special_tokens = encode_special_tokens,
            return_offsets = True,
            add_bos = False)

        output = EncodeOutputDict(
            ids = tokens,
            mask = torch.ones_like(tokens, device=self.sync.config['torch_device']),
            position_offsets = position_offsets
        )
        
        return output
        

    def decode(self, inputs: torch.Tensor, skip_special_tokens=True, logits_mode=False) -> Union[List[str], List[List[str]]]:
        """
        decoded tokens to strings

        input: 1D or 2D tensor containing tokens
        output: 2D list of strings
        """
        
        assert (inputs.ndim == 2 or inputs.ndim == 1), f"inputs need to be 1D or 2D tensor (batch, tokens), got: {inputs.shape}"

        
        # make sure inputs tensor is 2D
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        decoded_strings = []
        if logits_mode:
            for entry in inputs:
                decoded_strings.append([])
                for sample in entry:
                    decoded = self.tokenizer.decode(torch.tensor([sample]), decode_special_tokens=not skip_special_tokens)
                    decoded_strings[-1].append(decoded)
        else:
            for entry in inputs:
                decoded = self.tokenizer.decode(entry, decode_special_tokens=not skip_special_tokens)
                decoded_strings.append(decoded)
                
        if logits_mode:
            decoded_strings = decoded_strings[0]

        return decoded_strings
        

        
    
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

        # prompts = self.decode(inputs, skip_special_tokens=False)



        gen_settings = ExLlamaV2Sampler.Settings(
            temperature = 1.0, 
            token_repetition_penalty = 1.0,
            top_p = 0.0,
            top_k = 1
        )
        

        for idx, inputs_entry in enumerate(inputs):
            job = ExLlamaV2DynamicJob(
                input_ids = inputs_entry.unsqueeze(0).cpu(), # todo make it so no need for moving to cpu
                # min_new_tokens = self.sync.dhold.max_tokens_this_gen,
                max_new_tokens = self.sync.dhold.max_tokens_this_gen,
                stop_conditions = self.stop_token.tolist(), # if self.sync.dhold.inputs['beam_config']['use_beam_search'] else None,
                gen_settings = gen_settings,
                identifier = idx,
                return_top_tokens = self.sync.dhold.inputs['max_num_beams'],
                return_logits = True,
                decode_special_tokens=True
                # return_probs = True,
            )
            self.generator.enqueue(job)


        # Somewhere to store the streaming results
        collected_outputs = [""] * inputs.shape[-1]

        logits_merker = {}
        text_merker = []
        tokens_merker = []
        
        job_counter = 0
        while self.generator.num_remaining_jobs():
            stop_generation = False
            job_counter += 1
                        
            results = self.generator.iterate()
            results = [result for result in results if result['stage'] == "streaming"]

            for result in results:
                print(result)
                serial = result['serial']
                if serial not in logits_merker:
                    logits_merker[serial] = []
                if "logits" in result:
                    logits_merker[serial].append(result['logits'])
                elif result['eos_reason'] == "stop_token":
                    eos_token_logits = torch.zeros((1, 1, 128256))
                    eos_token_logits[0, 0, self.tokenizer.eos_token_id] = 1
                    logits_merker[serial].append(eos_token_logits)


        for key, val in logits_merker.items():
            print(f"{key}: {len(val)}")
            for entry in val:
                try:
                    print("  ", entry.shape)
                except:
                    print("  ", entry)

        

        self.generator.active_jobs = []

        # get all logits outputs to the same length so they can be concatenated in case one of them got stopped early
        # beams = [[] for _ in range(inputs.shape[0])]
        print(logits_merker)
        # if len(logits_merker) == 1 and next(iter(logits_merker.values())) == []:
            

        # if self.sync.dhold.inputs['beam_config']['use_beam_search']:
        beams = [torch.concatenate(val, dim=-2) for key, val in logits_merker.items()]

        print("number of beams:", len(beams))
        for index, beam in enumerate(beams):
            print(f"beam {index}:")
            for entry in beam:
                print(entry.shape)

        # truncate all beams to the shortest beam length
        min_length = beams[0].shape[-2]
        for index, beam in enumerate(beams):
            if beams[index].shape[-2] < min_length:
                min_length = beams[index].shape[-2]

        for index, beam in enumerate(beams):
            if beams[index].shape[-2] > min_length:
                beams[index] = beams[index][:, :min_length, :]
        
        logits = torch.concatenate(beams, dim=0)
        print(logits.shape)

        top_logits = find_top_indexes(self.sync, logits)
        



        tokens_decoded_merker = self.decode(top_logits[:, :, 0, 0].to(torch.int32))
        output_shape = logits.shape[:2]
        
        
        output = GenerateOutputDict(
            decoded_output = [["".join(entry) for entry in tokens_decoded_merker]],
            output_shape = output_shape,
            logits = None,
            top_logits = top_logits
        )



        

        return output        
