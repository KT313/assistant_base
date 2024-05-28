from .imports import *
from .model_holder import ModelHolder
from .data_holder import DataHolder
from .misc import softmax, find_top_indexes
from .processor_helper import ProcessorHelper

class Sync():
    def __init__(self, config=None):
        self.mhold = None
        self.dhold = None
        self.phelp = ProcessorHelper()
        self.config = config

    def prep_gen_inputs(self):
        args = self.dhold.inputs
        self.dhold.gen_inputs = args

        # check if model needs to be changed
        self.phelp.load_model(self)
        

        self.phelp.load_beam_config(self)
        self.phelp.build_prompt_string(self)
        self.phelp.prepare_model_generation_args(self)

        

    def get_best_path(self):

        args = self.dhold.gen_inputs

        total_probs  = []
        prediction_paths_probs = []
        prediction_paths_indices = []
        skip_path = []

        self.dhold.logits_merker = copy.deepcopy(self.dhold.logits)
        self.dhold.considered_tokens_num_merker = copy.deepcopy(self.dhold.considered_tokens_num)

        tokens = None
        try:
            tokens = args['tokens'].input_ids
        except:
            tokens = args['tokens']

        was_list = False
        if isinstance(tokens, list):
            was_list = True
            tokens = torch.tensor(tokens, device=self.config['torch_device'])
            
        batched_input_tokens = torch.concatenate((tokens.repeat(self.dhold.considered_tokens_num[0], 1), torch.tensor(self.dhold.logits[0, 0, :self.dhold.considered_tokens_num[0], 0], device=self.config['torch_device']).unsqueeze(1)), dim=-1).to(torch.long)
        batched_input_masks = torch.ones_like(batched_input_tokens, device=self.config['torch_device'])

        if self.mhold.current_model == "llama3-llava-next-8b" and len(args['images']) > 0:
            self.do_inference(limit_tokens=args['depth_beams'], alternative_input=batched_input_tokens, alternative_mask=torch.ones_like(batched_input_tokens[0].unsqueeze(0), device=self.config['torch_device']), llama_sequencial_batch=True)
        elif was_list:
            self.do_inference(limit_tokens=args['depth_beams'], alternative_input=batched_input_tokens.tolist(), llama_sequencial_batch=True)
        else:
            self.do_inference(limit_tokens=args['depth_beams'], alternative_input=batched_input_tokens, alternative_mask=batched_input_masks)

        

        for i in range(self.dhold.considered_tokens_num[0]):
            # case considered token is stop token:
            if np.any(self.dhold.logits_merker[0, 0, i, 0] == self.mhold.stop_token):
                total_probs.append(math.log(self.dhold.logits_merker[0, 0, i, 1]))
                prediction_paths_probs.append([math.log(self.dhold.logits[0, 0, i, 1])])
                prediction_paths_indices.append([self.dhold.logits[0, 0, i, 0]])
                skip_path.append(i)
                continue
                
            highest_path_probs = []
            highest_path_indices = []
            for token_num in range(self.dhold.logits.shape[1]):
                highest_path_probs.append(math.log(self.dhold.logits[i, token_num, 0, 1]))
                highest_path_indices.append(self.dhold.logits[i, token_num, 0, 0])
                pass
            total_prob = math.log(self.dhold.logits_merker[0, 0, i, 1])
            total_prob += sum(highest_path_probs)
            total_probs.append(total_prob)
            prediction_paths_probs.append([math.log(self.dhold.logits_merker[0, 0, i, 1])]+highest_path_probs)
            prediction_paths_indices.append([self.dhold.logits_merker[0, 0, i, 0]]+highest_path_indices)

        if args['debugmode']:
            print("paths total probs:", [round(entry, 3) for entry in total_probs])

        best_beam = max(enumerate(total_probs),key=lambda x: x[1])[0]

        self.dhold.best_beam_probs = prediction_paths_probs[best_beam]
        self.dhold.best_beam_indices = prediction_paths_indices[best_beam]

        

    def do_inference(self, limit_tokens=None, alternative_input=None, alternative_mask=None, llama_sequencial_batch=False):
        self.dhold.start_time_inference = time.time()
        if self.dhold.error:
            return None
        args = self.dhold.gen_inputs
        if limit_tokens != None:
            max_tokens_this_gen = limit_tokens
        else:
            max_tokens_this_gen = args['max_new_tokens']

        generated_tokens = 0
        self.mhold.stop_token = None
        gen_kwargs = {
            'max_new_tokens': max_tokens_this_gen,
            'do_sample': False,
            'temperature': 1,
            'output_scores': True,
            'return_dict_in_generate': True,
        }
        got_input_shape_already = True
        

        
        if args['model'] in ["llama3-llava-next-8b", "Hermes-2-Theta-Llama-3-8B", "phi-3-vision-128k-instruct"]:
            gen_function = self.mhold.model.generate
            get_logits = lambda output: find_top_indexes([token_logits.detach().cpu().numpy() for token_logits in output.scores], n_top=args['max_num_beams'])

        if args['model'] in ["llama3-llava-next-8b", "Hermes-2-Theta-Llama-3-8B"]:
            original_input_len = args['tokens'].shape[-1]
            attn_mask = torch.ones_like(args['tokens'], device=self.config['torch_device'])
            gen_kwargs.update({
                'num_beams': 1,
                'attention_mask': torch.ones_like(args['tokens'], device=self.config['torch_device']),
                'pad_token_id': self.mhold.tokenizer.eos_token_id,
            })
            gen_input = args['tokens']
            self.mhold.stop_token = self.mhold.tokenizer.eos_token_id
        
        if args['model'] == "llama3-llava-next-8b":
            gen_kwargs.update({
                'images': args['image_tensor'],
                'image_sizes': args['image_sizes'],
            })
            output_processor = lambda output: [self.mhold.tokenizer.decode(output.sequences[i][:], skip_special_tokens=True) for i in range(len(output.sequences))]
            shape_attr = lambda output: output.sequences[0][:].shape

        
            
        if args['model'] == "Hermes-2-Theta-Llama-3-8B":
            output_processor = lambda output: [self.mhold.tokenizer.decode(output.sequences[i][original_input_len:], skip_special_tokens=True) for i in range(len(output.sequences))]
            shape_attr = lambda output: output.sequences[0][original_input_len:].shape

        if args['model'] == "phi-3-vision-128k-instruct":
            gen_kwargs.update({
                'attention_mask': torch.ones_like(args['tokens'].input_ids, device=self.config['torch_device']),
                'pixel_values': args['tokens'].pixel_values if "pixel_values" in args['tokens'] else None,
                'image_sizes': args['tokens'].image_sizes if "image_sizes" in args['tokens'] else None,
                'eos_token_id': self.mhold.processor.tokenizer.eos_token_id
            })
            self.mhold.stop_token = self.mhold.processor.tokenizer.eos_token_id
            gen_input = args['tokens'].input_ids
            output_processor = lambda output: [self.mhold.processor.decode(output.sequences[i][args['tokens']['input_ids'].shape[1]:], skip_special_tokens=True) for i in range(len(output.sequences))]
            shape_attr = lambda output: output.sequences[:, args['tokens']['input_ids'].shape[1]:].shape
            
        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
            del gen_kwargs['max_new_tokens']
            del gen_kwargs['do_sample']
            del gen_kwargs['output_scores']
            del gen_kwargs['return_dict_in_generate']
            gen_kwargs.update({
                'max_tokens': max_tokens_this_gen,
                'stop': ["<|eot_id|>"],
                'echo': False,
                'top_k': 1,
                'logprobs': -1,
            })
            self.mhold.stop_token = self.mhold.model.tokenize("<|eot_id|>".encode('UTF-8'), special=True)
            got_input_shape_already = False
            gen_function = self.mhold.model
            gen_input = args['tokens']
            output_processor = lambda output: [out['text'] for out in output['choices']]
            shape_attr = lambda output: [1, output['usage']['completion_tokens']]
            input_shape_attr = lambda output: [1, output['usage']['prompt_tokens']]
            get_logits = lambda scores: find_top_indexes(self.mhold.model._scores[-self.dhold.output_shape[-1]:], args['max_num_beams'])

        if self.mhold.stop_token == None:
            raise Error('did/could not assign stop token')

        if alternative_input != None:
            gen_input = alternative_input

        if alternative_mask != None:
            gen_kwargs.update({
                'attention_mask': alternative_mask,
            })

        if not llama_sequencial_batch:
            gen_output = gen_function(gen_input, **gen_kwargs)

            self.dhold.returned_content = [entry.strip() for entry in output_processor(gen_output)]
            self.dhold.output_shape = getattr(gen_output, shape_attr) if isinstance(shape_attr, str) else shape_attr(gen_output)
            self.dhold.logits = get_logits(gen_output)
        else:
            returned_content = []
            output_shape = []
            logits = []
            for entry in gen_input:
                if isinstance(entry, torch.Tensor) and len(entry.shape) == 1:
                    entry = entry.unsqueeze(0)
                
                gen_output = gen_function(entry, **gen_kwargs)
    
                returned_content.append([entry.strip() for entry in output_processor(gen_output)])
                output_shape.append(getattr(gen_output, shape_attr) if isinstance(shape_attr, str) else shape_attr(gen_output))
                logits.append(get_logits(gen_output))
            self.dhold.returned_content = returned_content
            self.dhold.output_shape = np.array(output_shape)
            self.dhold.logits = np.concatenate(logits, axis=0)

        # get number of considered tokens for each batch
        merker = [1 for _ in range(self.dhold.logits.shape[0])] #  add the first one by default
        print(self.dhold.logits, self.dhold.logits.shape)
        for batch_num in range(self.dhold.logits.shape[0]):
            for top_logit_num in range(1, self.dhold.logits.shape[2]):
                if self.dhold.logits[batch_num][0][top_logit_num][1] >= args['min_conf_for_consider']:
                    merker[batch_num] += 1
                else: 
                    break
                if np.sum(self.dhold.logits[batch_num][0][:merker[batch_num]][0]) >= args['prob_sum_for_search']:
                    break
        self.dhold.considered_tokens_num = np.array(merker)


        
        
        if args['debugmode']: print("\n\nself.dhold.returned_content:", self.dhold.returned_content, "\n\n")



    # sets dhold.returned_content, dhold.output_shape, self.dhold.logits (and maybe dhold.input_shape)
    def generate(self):
        
        # if normal
        # generate no limit
        args = self.dhold.gen_inputs
        if not args['beam_config']['use_beam_search']:
            self.do_inference()
        # 
        else:
            self.dhold.generated_tokens = 0
            while self.dhold.generated_tokens < args['max_new_tokens']:
                # generate limit 1 token
                self.do_inference(limit_tokens=1)
        
                if self.dhold.considered_tokens_num[0] == 1:
                    self.dhold.tokens_to_add = [self.dhold.logits[0, 0, 0, 0]]
                    self.dhold.best_path_indices = self.dhold.tokens_to_add
                    self.dhold.additional_sure_tokens = 0
                    self.dhold.logits_merker = copy.deepcopy(self.dhold.logits)
                    self.dhold.considered_tokens_num_merker = copy.deepcopy(self.dhold.considered_tokens_num)
                    self.dhold.best_beam_indices = [self.dhold.logits[0, 0, 0, 0]]
                    
                else:
                    self.get_best_path()
        
                    self.dhold.tokens_to_add = [self.dhold.best_beam_indices[0]] # at least at the init token for the best path
                    self.dhold.additional_sure_tokens = 0
                    for i in range(1, len(self.dhold.best_beam_indices)): # skip 0 since already added
                        if self.dhold.best_beam_probs[i] >= math.log(args['min_conf_for_sure']):
                            self.dhold.additional_sure_tokens += 1
                            self.dhold.tokens_to_add.append(self.dhold.best_beam_indices[i])
                        else:
                            break
                        
                self.dhold.generated_tokens += len(self.dhold.tokens_to_add)
        
                was_input_ids = False
                try:
                    tokens = args['tokens'].input_ids
                    was_input_ids = True
                except:
                    tokens = args['tokens']
                    
                was_list = False
                if isinstance(tokens, list):
                    was_list = True
                    tokens = torch.tensor(tokens, device=self.config['torch_device']).unsqueeze(0)

                if was_input_ids:
                    args['tokens'].input_ids = torch.concatenate((tokens, torch.tensor(self.dhold.tokens_to_add, device=self.config['torch_device']).to(torch.long).unsqueeze(0)), dim=-1)
                    attn_mask = torch.ones_like(tokens, device=self.config['torch_device'])
                    
                else:
                    args['tokens'] = torch.concatenate((tokens, torch.tensor(self.dhold.tokens_to_add, device=self.config['torch_device']).to(torch.long).unsqueeze(0)), dim=-1)
                    attn_mask = torch.ones_like(tokens, device=self.config['torch_device'])
                    if was_list:
                        args['tokens'] = args['tokens'].tolist()[0]
                
        
                if args['debugmode']:
                    print(" | ".join([str(round(entry, 5)).ljust(14) for entry in self.dhold.logits_merker[0, 0, :, 1]]))
                    if was_list:
                        print(" | ".join([self.mhold.model.detokenize([int(entry)]).decode('UTF-8').strip().ljust(14) for entry in self.dhold.logits_merker[0, 0, :, 0]]))
                    else:
                        try:
                            print(" | ".join([self.mhold.tokenizer.decode([int(entry)], skip_special_tokens=False, clean_up_tokenization_space=True).strip().ljust(14) for entry in self.dhold.logits_merker[0, 0, :, 0]]))
                        except:
                            print(" | ".join([self.mhold.processor.tokenizer.decode([int(entry)], skip_special_tokens=False, clean_up_tokenization_space=True).strip().ljust(14) for entry in self.dhold.logits_merker[0, 0, :, 0]]))
                    if np.any(self.dhold.considered_tokens_num_merker == 1):
                        print("-> single considered token, not doing beam search")
                    else:
                        print(f"-> using {self.dhold.considered_tokens_num_merker} beams")
        
                    print("\n")
                    if was_list:
                        print(f"current generation: {self.mhold.model.detokenize(args['tokens'][self.dhold.original_input_len:-len(self.dhold.tokens_to_add)]).decode('UTF-8')}\x1b[32m{self.mhold.model.detokenize([int(a) for a in self.dhold.tokens_to_add]).decode('UTF-8')}\x1b[0m \x1b[37m{self.mhold.model.detokenize([int(a) for a in self.dhold.best_beam_indices[1+self.dhold.additional_sure_tokens:]]).decode('UTF-8')}\x1b[0m") # \[90m or \[37m for gray \x1b[43
                    else:
                        try:
                            print(f"current generation: {self.mhold.tokenizer.decode(args['tokens'][0][self.dhold.original_input_len:-len(self.dhold.tokens_to_add)], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[32m{self.mhold.tokenizer.decode([int(a) for a in self.dhold.tokens_to_add], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m \x1b[37m{self.mhold.tokenizer.decode([int(a) for a in self.dhold.best_beam_indices[1+self.dhold.additional_sure_tokens:]], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m") # \[90m or \[37m for gray \x1b[43
                        except:
                            print(f"current generation: {self.mhold.processor.tokenizer.decode(args['tokens'].input_ids[0][self.dhold.original_input_len:-len(self.dhold.tokens_to_add)], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[32m{self.mhold.processor.tokenizer.decode([int(a) for a in self.dhold.tokens_to_add], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m \x1b[37m{self.mhold.processor.tokenizer.decode([int(a) for a in self.dhold.best_beam_indices[1+self.dhold.additional_sure_tokens:]], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m") # \[90m or \[37m for gray \x1b[43
                    print("\n\n\n")
        
                if np.any(self.dhold.tokens_to_add == self.mhold.stop_token):
                    if args['debugmode']:
                        print("tokens to add contained stop token, stopping.")
                    break
                
                if self.dhold.generated_tokens >= args['max_new_tokens']:
                    reached_token_limit = True
                    if args['debugmode']:
                        print("reached max_new_tokens, stopping.")
                    break

            # end
            if was_list:
                self.dhold.returned_content = [self.mhold.model.detokenize(args['tokens'][self.dhold.original_input_len:]).decode('UTF-8')]
            else:
                try:
                    self.dhold.returned_content = [self.mhold.tokenizer.decode(args['tokens'].tolist()[0][self.dhold.original_input_len:], skip_special_tokens=True)]
                except:
                    self.dhold.returned_content = [self.mhold.processor.tokenizer.decode(args['tokens'].input_ids.tolist()[0][self.dhold.original_input_len:], skip_special_tokens=True)]


    
    def make_new_dhold(self):
        self.dhold = DataHolder()

    def make_new_mhold(self):
        self.mhold = ModelHolder()