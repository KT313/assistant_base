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


        
        
    def encode(self, inputs: Union[str, List[str]], encode_special_tokens=True) -> EncodeOutputDict:
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
        print("EncodeOutputDict:", output)
        
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
            print("encoder inputs:", inputs)
            for entry in inputs:
                print("encoder inputs entry:", entry)
                decoded = self.tokenizer.decode(entry, decode_special_tokens=not skip_special_tokens)
                print("encoder inputs entry decoded:", decoded)
                decoded_strings.append(decoded)
                
        if logits_mode:
            decoded_strings = decoded_strings[0]

        return decoded_strings
        
        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if logits_mode:
            decoded = [self.tokenizer.batch_decode(entry, skip_special_tokens=skip_special_tokens) for entry in inputs]
        else:
            decoded = [self.tokenizer.batch_decode(inputs, skip_special_tokens=skip_special_tokens)]
        
        if logits_mode:
            decoded = decoded[0]

        print("decoded:", decoded)
        exit()

        return decoded
        """
        
    
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
                max_new_tokens = self.sync.dhold.max_tokens_this_gen,
                stop_conditions = [self.tokenizer.eos_token_id],
                gen_settings = gen_settings,
                identifier = idx,
                return_top_tokens = self.sync.dhold.inputs['max_num_beams'],
                return_logits = True,
                # return_probs = True,
            )
            self.generator.enqueue(job)


        # Somewhere to store the streaming results
        collected_outputs = [""] * inputs.shape[-1]

        logits_merker = []
        
        job_counter = 0
        while self.generator.num_remaining_jobs():
            print(f"job {job_counter}:")
            job_counter += 1
            
            merker = []
            
            results = self.generator.iterate()
            print("results:", results)
            
            for result in results:
                if "logits" in result:
                    merker.append(result['logits'])
            print(f"merker len: {len(merker)}")
            logits_merker.append(torch.concatenate(merker, dim=0))

        self.generator.active_jobs = []
            
        # self.generator.reset_page_table()
        # self.generator.cancel()

        print("logits merker before concat:")
        for a in logits_merker:
            print(f"   {a.shape}")
        logits = torch.concatenate(logits_merker, dim=-2)
        print("logits after concat:", logits.shape)
        # logits = logits.permute(1, 0, 2) #  get into order batch, token, logits
        print("logits shape:", logits.shape)
        top_logits = find_top_indexes(self.sync, logits)
        print("top logits shape:", top_logits.shape)
        



        tokens_decoded_merker = self.decode(top_logits[:, :, 0, 0].to(torch.int32))
        print("tokens_decoded_merker:", tokens_decoded_merker)
        output_shape = logits.shape[:2]
        
        output = GenerateOutputDict(
            decoded_output = [["".join(entry) for entry in tokens_decoded_merker]],
            output_shape = output_shape,
            logits = None,
            top_logits = top_logits
        )



        

        print("output:", output)
        return output        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        """

        print("inputs shape:", inputs.shape)
        print("position_offsets shape:", kwargs['position_offsets'].shape)

        print("cache batch size:", self.cache.batch_size)
        if self.cache.batch_size != inputs.shape[0]:
            pretrained = self.sync.config['models'][self.sync.mhold.current_model]['path']
            config = ExLlamaV2Config(pretrained)
            self.model = ExLlamaV2(config)
            self.cache = ExLlamaV2Cache(self.model, max_seq_len = 4096, batch_size=inputs.shape[0], lazy = True)
            self.model.load_autosplit(self.cache, progress = True)
        print("cache batch size:", self.cache.batch_size)

        ####################################################

        num_tokens = kwargs['max_new_tokens']

        stop_token = self.tokenizer.eos_token_id
        random.seed(0)
        batch_size = inputs.shape[0]
        loras = None

        logits_merker = []
        tokens_decoded_merker = []
        self.sequence_ids = inputs
        
        for i in range(num_tokens):
            print(f"gen token {i}")

            # Truncate prompt if generation would cause cache overflow
            overflow = inputs.shape[-1] + num_tokens - self.model.config.max_seq_len
            if overflow > 0: inputs = inputs[:, overflow:]
            else: overflow = 0
    
            mask = torch.ones_like(inputs, device=inputs.device)
            first_token = inputs.shape[-1]
            position_offsets = None # kwargs['position_offsets']

            print("\n\n\n--------------------------\n")

            print("inputs:", inputs.shape, inputs.device, inputs)
            print("mask:", mask.shape, mask.device, mask)
            try:
                print("self.sequence_ids:", self.sequence_ids.shape, self.sequence_ids.device, self.sequence_ids)
            except:
                print("self.sequence_ids:", self.sequence_ids)
            try:
                print("position_offsets:", position_offsets.shape, position_offsets.device, position_offsets)
            except:
                print("position_offsets:", position_offsets)
    
            self._gen_begin_base(inputs,
                                 mask,
                                 loras,
                                 position_offsets = position_offsets,
                                 input_embeddings = None)
        
            logits = self.model.forward(
                self.sequence_ids[:, -1:],
                self.cache,
                input_mask = mask,
                loras = loras,
                position_offsets = kwargs['position_offsets'],
                indexed_embeddings = None
            ).float().cpu()

            print("logits shape before softmax:", logits.shape, logits[0, 0, :10])
            # logits = torch.softmax(logits, dim=-1)
            # print("logits shape after softmax:", logits.shape, logits[0, 0, :10])
            logits_merker.append(logits)
            top_token = find_top_indexes(self.sync, torch.tensor(logits))[:, 0, 0, 0]
            print("before concat:", inputs.shape, inputs.device, top_token.shape, top_token.device, top_token.to(torch.int32).to(self.sync.config['torch_device']).unsqueeze(1).shape)
            inputs = torch.concatenate([inputs, top_token.to(torch.int32).to(self.sync.config['torch_device']).unsqueeze(1)], dim=-1)
            top_token_decoded = self.sync.mhold.helper.decode(torch.tensor(top_token.to(torch.int32)), skip_special_tokens=False)[0]
            print(top_token_decoded)
            
            if torch.any(torch.isin(top_token.to(torch.int32), torch.tensor(self.sync.mhold.stop_token))): 
                break
            tokens_decoded_merker.append(top_token_decoded)

            self.sequence_ids = torch.cat([self.sequence_ids, top_token.to(torch.int32).to(self.sync.config['torch_device']).unsqueeze(1)], dim = 1)

        ####################################################

        tokens_decoded_merker = [tokens_decoded_merker]
        logits = torch.concatenate(logits_merker, dim=-2)
        print("tokens_decoded_merker:", tokens_decoded_merker)
        print("logits shape:", logits.shape)
        top_logits = find_top_indexes(self.sync, logits)
        print("top_logits shape:", top_logits.shape)
        print("tokens_decoded_merker:", tokens_decoded_merker)

        output_shape = logits.shape[:2]

        # exit()
        
        

        
        output = GenerateOutputDict(
            decoded_output = [["".join(entry) for entry in tokens_decoded_merker]],
            output_shape = output_shape,
            logits = None,
            top_logits = top_logits
        )
        
        return output

    def _gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           input_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]
            """