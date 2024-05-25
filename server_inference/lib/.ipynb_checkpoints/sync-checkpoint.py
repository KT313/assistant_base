from .imports import *
from .model_holder import ModelHolder
from .data_holder import DataHolder
from .misc import softmax, find_top_indexes

class Sync():
    def __init__(self, config=None):
        self.mhold = None
        self.dhold = None
        self.config = config

    def prep_gen_inputs(self):
        # args should contain:
        # model
        # chat
        # image

        args = self.dhold.inputs

        self.dhold.gen_inputs = args

        # check if model needs to be changed
        if self.mhold == None or args['model'] != self.mhold.current_model or args['model_dtype'] != self.mhold.current_dtype:
            self.mhold = ModelHolder()
            self.mhold.load_model(self, args['model'], args['model_dtype'])
        if args['debugmode']:
            print(args['model'], self.mhold.current_model, flush=True)

        # build prompt string
        args['chat'] = [chat for chat in args['chat'] if chat['role'] != "System"]
        for i in range(len(args['chat'])):
            old_role = args['chat'][i]['role']
            if old_role == "User":
                args['chat'][i]['role'] = "user"
            if old_role == "AI":
                args['chat'][i]['role'] = "assistant"

        if args['use_functions'] and "functions" in self.config['models'][self.mhold.current_model]:
            args['chat'].insert(0, {'role': 'system', 'content': self.config['models'][self.mhold.current_model]['functions']})
        elif 'manual_system_prompt' in args and args['manual_system_prompt'].strip() != "":
            args['chat'].insert(0, {'role': 'system', 'content': args['manual_system_prompt'].strip()})
        elif "system_prompt" in self.config['models'][self.mhold.current_model]:
            args['chat'].insert(0, {'role': 'system', 'content': self.config['models'][self.mhold.current_model]['system_prompt']})
        
        template_type = self.config['models'][args['model']]['template']
        template = self.config['chat_templates'][template_type]

        prompt_string = ""
        prompt_string += template['init']

        if template['roles as string']:
            for index, entry in enumerate(args['chat']):
                image_string = ""
                if index == (len(args['chat'])-1) and len(args['images']) > 0:
                    image_string = template['image token']
                prompt_string += f"{template['role start']}{entry['role']}{template['role end']}{image_string}{entry['content']}{template['end text']}"
            prompt_string += f"{template['role start']}assistant{template['role end']}"
        else:
            for index, entry in enumerate(args['chat']):
                image_string = ""
                if index == (len(args['chat'])-1) and len(args['images']) > 0:
                    image_string = template['image token']
                if entry['role'] == "system":
                    role_token = template['system role']
                elif entry['role'] == "user":
                    role_token = template['user role']
                elif entry['role'] == "assistant":
                    role_token = template['ai role']
                prompt_string += f"{role_token}{image_string}{entry['content']}{template['end text']}"
            prompt_string += f"{template['ai role']}"

        print(f"generated prompt string:\n\"{prompt_string}\"")

        if args['model'] == "llama3-llava-next-8b":
            image_tensor = None
            img_token_index = None
            image_sizes = None
            if len(args['images']) > 0:
                image_tensor = process_images(args['images'], self.mhold.image_processor, self.mhold.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=self.config['torch_device']) for _image in image_tensor]
                image_sizes = [image.size for image in args['images']]
                img_token_index = IMAGE_TOKEN_INDEX

                
            tokens = tokenizer_image_token(prompt_string, self.mhold.tokenizer, img_token_index, return_tensors="pt").unsqueeze(0).to(self.config['torch_device'])
            image_sizes = [image.size for image in args['images']]

            self.dhold.gen_inputs['tokens'] = tokens
            self.dhold.gen_inputs['image_tensor'] = image_tensor
            self.dhold.gen_inputs['image_sizes'] = image_sizes
            self.dhold.input_shape = self.dhold.gen_inputs['tokens'].shape

        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":

            self.dhold.gen_inputs['text'] = prompt_string

        if args['model'] == "Hermes-2-Theta-Llama-3-8B":

            tokens = self.mhold.tokenizer(prompt_string, return_tensors="pt").input_ids.to(self.config['torch_device'])
            
            self.dhold.gen_inputs['tokens'] = tokens
            self.dhold.gen_inputs['beam_config'] = args['beam_config']
            self.dhold.input_shape = self.dhold.gen_inputs['tokens'].shape

        if args['model'] == "phi-3-vision-128k-instruct":
            image_input = None
            if len(args['images']) > 0:
                image_input = args['images']
            
            tokens = self.mhold.processor(prompt_string, image_input, return_tensors="pt").to(self.config['torch_device'])

            self.dhold.gen_inputs['tokens'] = tokens
            self.dhold.input_shape = self.dhold.gen_inputs['tokens'].input_ids.shape

        self.dhold.gen_inputs['model'] = args['model']

    def get_best_path(self, args, considered_tokens_probs, considered_tokens_indices, stop_token):
        total_probs  = []
        prediction_paths_probs = []
        prediction_paths_indices = []
        skip_path = []

        batched_input_tokens = torch.concatenate((args['tokens'].repeat(len(considered_tokens_indices), 1), torch.tensor(considered_tokens_indices, device=self.config['torch_device']).unsqueeze(1)), dim=-1)
        batched_input_masks = torch.ones_like(batched_input_tokens, device=self.config['torch_device'])
        
        
        beam_output = self.mhold.model.generate(
            batched_input_tokens,
            attention_mask = batched_input_masks,
            max_new_tokens=args['depth_beams'],
            temperature=1.0,
            repetition_penalty=1.1,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.mhold.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id = stop_token
        )

        for i in range(len(considered_tokens_probs)):
            # case considered token is stop token:
            if considered_tokens_indices[i] == stop_token:
                total_probs.append(math.log(considered_tokens_probs[i]))
                prediction_paths_probs.append([math.log(considered_tokens_probs[i])])
                prediction_paths_indices.append([considered_tokens_indices[i]])
                skip_path.append(i)
                continue
                
            highest_path_probs = []
            highest_path_indices = []
            for token_num in range(len(beam_output.scores)):
                beam_probabilities, beam_indices = torch.topk(torch.softmax(beam_output.scores[token_num][i], dim=-1), k=args['max_num_beams'])
                highest_path_probs.append(math.log(beam_probabilities.tolist()[0]))
                highest_path_indices.append(beam_indices.tolist()[0])
            total_prob = math.log(considered_tokens_probs[i])
            total_prob += sum(highest_path_probs)
            total_probs.append(total_prob)
            prediction_paths_probs.append([math.log(considered_tokens_probs[i])]+highest_path_probs)
            prediction_paths_indices.append([considered_tokens_indices[i]]+highest_path_indices)

        if args['debugmode']:
            print("paths total probs:", [round(entry, 3) for entry in total_probs])

        best_beam = max(enumerate(total_probs),key=lambda x: x[1])[0]

        return prediction_paths_probs[best_beam], prediction_paths_indices[best_beam]

    def generate(self, limit_tokens=None):
        self.dhold.start_time_inference = time.time()
        if self.dhold.error:
            return None
        args = self.dhold.gen_inputs
        if limit_tokens != None:
            args['max_new_tokens'] = limit_tokens
        # args should contain:
        # model
        # gen_inputs

        # general settings

        generated_tokens = 0

        max_num_beams = int(args['beam_config']['max_num_beams'].strip())
        depth_beams = int(args['beam_config']['depth_beams'].strip())
        min_conf_for_sure = float(args['beam_config']['min_conf_for_sure'].strip())
        min_conf_for_consider = float(args['beam_config']['min_conf_for_consider'].strip())
        prob_sum_for_search = float(args['beam_config']['prob_sum_for_search'].strip())

        args['max_num_beams'] = max_num_beams
        args['depth_beams'] = depth_beams
        args['min_conf_for_sure'] = min_conf_for_sure
        args['min_conf_for_consider'] = min_conf_for_consider
        args['prob_sum_for_search'] = prob_sum_for_search

        gen_kwargs = {
            'max_new_tokens': args['max_new_tokens'],
            'do_sample': False,
            'temperature': 1,
            'output_scores': True,
            'return_dict_in_generate': True,
        }
        got_input_shape_already = True

        

        # model specific settings
        if not args['beam_config']['use_beam_search']:
            
            if args['model'] == "llama3-llava-next-8b":
                print(args['tokens'].shape)
                original_input_len = args['tokens'].shape[-1]
                attn_mask = torch.ones_like(args['tokens'], device=self.config['torch_device'])
                gen_kwargs.update({
                    'images': args['image_tensor'],
                    'image_sizes': args['image_sizes'],
                    'attention_mask': attn_mask,
                    'num_beams': 1,
                    'pad_token_id': self.mhold.tokenizer.eos_token_id,
                })
                gen_function = self.mhold.model.generate
                gen_input = args['tokens']
                # output_processor = lambda output: self.mhold.tokenizer.batch_decode(output, skip_special_tokens=True)
                output_processor = lambda output: [self.mhold.tokenizer.decode(output.sequences[i][:], skip_special_tokens=True) for i in range(len(output.sequences))]
                shape_attr = lambda output: output.sequences[0][:].shape
                get_logits = lambda output: find_top_indexes([token_logits.detach().cpu().numpy() for token_logits in output.scores], n_top=max_num_beams)
                
            if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
                del gen_kwargs['max_new_tokens']
                del gen_kwargs['do_sample']
                del gen_kwargs['output_scores']
                del gen_kwargs['return_dict_in_generate']
                gen_kwargs.update({
                    'max_tokens': args['max_new_tokens'],
                    'stop': ["<|eot_id|>", "<|end_of_text|>"],
                    'echo': False,
                    'top_k': 1,
                    'logprobs': -1,
                })
                got_input_shape_already = False
                gen_function = self.mhold.model
                gen_input = args['text']
                output_processor = lambda output: [out['text'] for out in output['choices']]
                shape_attr = lambda output: [1, output['usage']['completion_tokens']]
                input_shape_attr = lambda output: [1, output['usage']['prompt_tokens']]
                get_logits = lambda scores: find_top_indexes(self.mhold.model._scores[-self.dhold.output_shape[-1]:], max_num_beams)
                
            if args['model'] == "Hermes-2-Theta-Llama-3-8B":
                original_input_len = args['tokens'].shape[-1]
                attn_mask = torch.ones_like(args['tokens'], device=self.config['torch_device'])
                
                gen_kwargs.update({
                    'attention_mask': attn_mask,
                    'num_beams': 1,
                    'eos_token_id': self.mhold.tokenizer.eos_token_id,
                    'output_scores': True,
                    'return_dict_in_generate': True,
                    'pad_token_id': 128003
                })
                gen_function = self.mhold.model.generate
                gen_input = args['tokens']
                output_processor = lambda output: [self.mhold.tokenizer.decode(output.sequences[i][original_input_len:], skip_special_tokens=True) for i in range(len(output.sequences))]
                shape_attr = lambda output: output.sequences[0][original_input_len:].shape
                get_logits = lambda output: find_top_indexes([token_logits.detach().cpu().numpy() for token_logits in output.scores], n_top=max_num_beams)

            if args['model'] == "phi-3-vision-128k-instruct":
                gen_kwargs.update({
                    'attention_mask': args['tokens'].attention_mask,
                    'pixel_values': args['tokens'].pixel_values if "pixel_values" in args['tokens'] else None,
                    'image_sizes': args['tokens'].image_sizes if "image_sizes" in args['tokens'] else None,
                    'eos_token_id': self.mhold.processor.tokenizer.eos_token_id
                })
                gen_function = self.mhold.model.generate
                gen_input = args['tokens'].input_ids
                output_processor = lambda output: [self.mhold.processor.decode(output.sequences[i][args['tokens']['input_ids'].shape[1]:], skip_special_tokens=True) for i in range(len(output.sequences))]
                shape_attr = lambda output: output.sequences[:, args['tokens']['input_ids'].shape[1]:].shape
                get_logits = lambda output: find_top_indexes([token_logits.detach().cpu().numpy() for token_logits in output.scores], n_top=max_num_beams)
    
            
            gen_output = gen_function(gen_input, **gen_kwargs)

            self.dhold.returned_content = [entry.strip() for entry in output_processor(gen_output)]
            self.dhold.output_shape = getattr(gen_output, shape_attr) if isinstance(shape_attr, str) else shape_attr(gen_output)            
            
            self.dhold.logits = get_logits(gen_output)
            print(f"logits:\n{self.dhold.logits}")
            # for ls in self.dhold.logits:
            #     for val in ls:
            #         print(val[0], self.mhold.tokenizer.decode(val[0]))
            #     print()
            
            if not got_input_shape_already:
                self.dhold.input_shape = input_shape_attr(gen_output)
        
            if args['debugmode']:
                print("\n\nself.dhold.returned_content:", self.dhold.returned_content, "\n\n")

        else:
            pass


        return None










        
        
        if args['model'] == "Hermes-2-Theta-Llama-3-8B":

            original_input_len = args['tokens'].shape[-1]
            attn_mask = torch.ones_like(args['tokens'], device=self.config['torch_device'])

            reached_token_limit = False

            num_beams_this_run = max_num_beams
            
            if args['debugmode']:
                print("input:", self.mhold.tokenizer.decode(args['tokens'][0], skip_special_tokens=False, clean_up_tokenization_space=True))

            
            
            for step in range(args['max_new_tokens']):

                # custom beam search                
                gen_output = self.mhold.model.generate(
                    args['tokens'],
                    attention_mask = attn_mask,
                    max_new_tokens=1,
                    temperature=1.0,
                    repetition_penalty=1.1,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.mhold.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id = 128003
                )
    
                probabilities, indices = torch.topk(torch.softmax(gen_output.scores[0].detach(), dim=-1), k=8)
                considered_tokens_probs = []
                considered_tokens_indices = []
                for i in range(max_num_beams):
                    if probabilities[0].tolist()[i] >= args['min_conf_for_consider']:
                        if step == 0 and indices[0].tolist()[i] == 128003:
                            continue
                        considered_tokens_probs.append(probabilities[0].tolist()[i])
                        considered_tokens_indices.append(indices[0].tolist()[i])
                    if sum(considered_tokens_probs) >= prob_sum_for_search:
                        break

                if len(considered_tokens_indices) == 1:
                    tokens_to_add = [considered_tokens_indices[0]]
                    best_path_indices = tokens_to_add
                    additional_sure_tokens = 0
                    
                else:
                    best_path_probs, best_path_indices = self.get_best_path(args, considered_tokens_probs, considered_tokens_indices, stop_token=128003)
        
                    tokens_to_add = [best_path_indices[0]] # at least at the init token for the best path
                    additional_sure_tokens = 0
                    for i in range(1, len(best_path_indices)): # skip 0 since already added
                        if best_path_probs[i] >= math.log(min_conf_for_sure):
                            additional_sure_tokens += 1
                            tokens_to_add.append(best_path_indices[i])
                        else:
                            break
                        
                generated_tokens += len(tokens_to_add)
                
                args['tokens'] = torch.concatenate((args['tokens'], torch.tensor(tokens_to_add, device=self.config['torch_device']).unsqueeze(0)), dim=-1)
                attn_mask = torch.ones_like(args['tokens'], device=self.config['torch_device'])

                if args['debugmode']:
                    print(" | ".join([str(round(entry, 5)).ljust(14) for entry in probabilities[0].tolist()]))
                    print(" | ".join([self.mhold.tokenizer.decode(entry, skip_special_tokens=False, clean_up_tokenization_space=True).strip().ljust(14) for entry in indices[0].tolist()]))
                    if len(considered_tokens_indices) == 1:
                        print("-> single considered token, not doing beam search")
                    else:
                        print(f"-> using {len(considered_tokens_probs)} beams")
    
                    print("\n")
                    print(f"current generation: {self.mhold.tokenizer.decode(args['tokens'][0][original_input_len:-len(tokens_to_add)], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[32m{self.mhold.tokenizer.decode(tokens_to_add, skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m \x1b[37m{self.mhold.tokenizer.decode(best_path_indices[1+additional_sure_tokens:], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m") # \[90m or \[37m for gray \x1b[43
                    print("\n\n\n")

                if 128003 in tokens_to_add:
                    if args['debugmode']:
                        print("tokens to add contained stop token, stopping.")
                    break
                
                if generated_tokens >= args['max_new_tokens']:
                    reached_token_limit = True
                    if args['debugmode']:
                        print("reached max_new_tokens, stopping.")
                    break

            response = self.mhold.tokenizer.decode(args['tokens'][0][original_input_len:], skip_special_tokens=True, clean_up_tokenization_space=True)
            if reached_token_limit:
                response += "<reached_token_limit>"
            self.dhold.output_shape = args['tokens'][0][original_input_len:].shape
            self.dhold.returned_content = [response.strip()]

            if args['debugmode']:
                print(f"response: {self.dhold.returned_content[0]}")
            
                print("\n\n\n")

    

    def make_new_dhold(self):
        self.dhold = DataHolder()