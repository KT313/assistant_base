from .imports import *
from .misc import softmax, find_top_indexes

class ProcessorHelper():
    def __init__(self):
        pass

    def load_model(self, sync):
        args = sync.dhold.inputs
        if sync.mhold == None or args['model'] != sync.mhold.current_model or args['model_dtype'] != sync.mhold.current_dtype:
            sync.make_new_mhold()
            sync.mhold.load_model(sync, args['model'], args['model_dtype'])
        if args['debugmode']: print(args['model'], sync.mhold.current_model, flush=True)
    
    def build_prompt_string(self, sync):
        args = sync.dhold.inputs
        args['chat'] = [chat for chat in args['chat'] if chat['role'] != "System"]
        for i in range(len(args['chat'])):
            old_role = args['chat'][i]['role']
            if old_role == "User": args['chat'][i]['role'] = "user"
            if old_role == "AI": args['chat'][i]['role'] = "assistant"

        if args['use_functions'] and "functions" in sync.config['models'][sync.mhold.current_model]:
            args['chat'].insert(0, {'role': 'system', 'content': sync.config['models'][sync.mhold.current_model]['functions']})
        elif 'manual_system_prompt' in args and args['manual_system_prompt'].strip() != "":
            args['chat'].insert(0, {'role': 'system', 'content': args['manual_system_prompt'].strip()})
        elif "system_prompt" in sync.config['models'][sync.mhold.current_model]:
            args['chat'].insert(0, {'role': 'system', 'content': sync.config['models'][sync.mhold.current_model]['system_prompt']})
        
        template_type = sync.config['models'][args['model']]['template']
        template = sync.config['chat_templates'][template_type]

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

        sync.dhold.prompt_string = prompt_string
        print(f"built prompt string:\n\"{sync.dhold.prompt_string}\"")

    def load_beam_config(self, sync):
        args = sync.dhold.inputs
        
        args['max_num_beams'] = int(args['beam_config']['max_num_beams'].strip())
        args['depth_beams'] = int(args['beam_config']['depth_beams'].strip())
        args['min_conf_for_sure'] = float(args['beam_config']['min_conf_for_sure'].strip())
        args['min_conf_for_consider'] = float(args['beam_config']['min_conf_for_consider'].strip())
        args['prob_sum_for_search'] = float(args['beam_config']['prob_sum_for_search'].strip())

    def prepare_model_generation_args(self, sync):
        args = sync.dhold.inputs
        if args['model'] == "llama3-llava-next-8b":
            image_tensor = None
            img_token_index = None
            image_sizes = None
            if len(args['images']) > 0:
                image_tensor = process_images(args['images'], sync.mhold.image_processor, sync.mhold.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=sync.config['torch_device']) for _image in image_tensor]
                image_sizes = [image.size for image in args['images']]
                img_token_index = IMAGE_TOKEN_INDEX

                
            tokens = tokenizer_image_token(sync.dhold.prompt_string, sync.mhold.tokenizer, img_token_index, return_tensors="pt").unsqueeze(0).to(sync.config['torch_device'])
            image_sizes = [image.size for image in args['images']]

            sync.dhold.gen_inputs['tokens'] = tokens
            sync.dhold.gen_inputs['image_tensor'] = image_tensor
            sync.dhold.gen_inputs['image_sizes'] = image_sizes
            if image_tensor != None:
                sync.dhold.input_shape = sync.dhold.gen_inputs['tokens'].shape #, sync.dhold.gen_inputs['image_tensor'][0].shape] TODO: find out how many tokens / embeddings one image is equal to
            else:
                sync.dhold.input_shape = sync.dhold.gen_inputs['tokens'].shape
            sync.dhold.original_input_len = sync.dhold.gen_inputs['tokens'].shape[-1]

        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":

            sync.dhold.gen_inputs['tokens'] = sync.mhold.model.tokenize(sync.dhold.prompt_string.encode('UTF-8'))
            sync.dhold.input_shape = [1, len(sync.dhold.gen_inputs['tokens'])]
            sync.dhold.original_input_len = len(sync.dhold.gen_inputs['tokens'])

        if args['model'] == "Hermes-2-Theta-Llama-3-8B":

            tokens = sync.mhold.tokenizer(sync.dhold.prompt_string, return_tensors="pt").input_ids.to(sync.config['torch_device'])
            
            sync.dhold.gen_inputs['tokens'] = tokens
            sync.dhold.gen_inputs['beam_config'] = args['beam_config']
            sync.dhold.input_shape = sync.dhold.gen_inputs['tokens'].shape
            sync.dhold.original_input_len = sync.dhold.gen_inputs['tokens'].shape[-1]

        if args['model'] == "phi-3-vision-128k-instruct":
            image_input = None
            if len(args['images']) > 0:
                image_input = args['images']
            
            tokens = sync.mhold.processor(sync.dhold.prompt_string, image_input, return_tensors="pt").to(sync.config['torch_device'])

            sync.dhold.gen_inputs['tokens'] = tokens
            sync.dhold.input_shape = sync.dhold.gen_inputs['tokens'].input_ids.shape
            sync.dhold.original_input_len = sync.dhold.gen_inputs['tokens'].input_ids.shape[-1]

        sync.dhold.gen_inputs['model'] = args['model']

    def append_tokens_to_add_to_tokens(self, sync):
        args = sync.dhold.inputs
        was_input_ids = False
        try:
            tokens = args['tokens'].input_ids
            was_input_ids = True
        except:
            tokens = args['tokens']
            
        sync.dhold.was_list = False
        if isinstance(tokens, list):
            sync.dhold.was_list = True
            tokens = torch.tensor(tokens, device=sync.config['torch_device']).unsqueeze(0)

        if was_input_ids:
            args['tokens'].input_ids = torch.concatenate((tokens, torch.tensor(sync.dhold.tokens_to_add, device=sync.config['torch_device']).to(torch.long).unsqueeze(0)), dim=-1)
            
        else:
            args['tokens'] = torch.concatenate((tokens, torch.tensor(sync.dhold.tokens_to_add, device=sync.config['torch_device']).to(torch.long).unsqueeze(0)), dim=-1)
            if sync.dhold.was_list:
                args['tokens'] = args['tokens'].tolist()[0]

    def print_beam_debug_info(self, sync):
        args = sync.dhold.inputs
        if args['debugmode']:
            print(" | ".join([str(round(entry, 5)).ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 1]]))
            if sync.dhold.was_list:
                print(" | ".join([sync.mhold.model.detokenize([int(entry)]).decode('UTF-8').strip().ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 0]]))
            else:
                try:
                    print(" | ".join([sync.mhold.tokenizer.decode([int(entry)], skip_special_tokens=False, clean_up_tokenization_space=True).strip().ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 0]]))
                except:
                    print(" | ".join([sync.mhold.processor.tokenizer.decode([int(entry)], skip_special_tokens=False, clean_up_tokenization_space=True).strip().ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 0]]))
            if np.any(sync.dhold.considered_tokens_num_merker == 1):
                print("-> single considered token, not doing beam search")
            else:
                print(f"-> using {sync.dhold.considered_tokens_num_merker} beams")

            print("\n")
            if sync.dhold.was_list:
                print(f"current generation: {sync.mhold.model.detokenize(args['tokens'][sync.dhold.original_input_len:-len(sync.dhold.tokens_to_add)]).decode('UTF-8')}\x1b[32m{sync.mhold.model.detokenize([int(a) for a in sync.dhold.tokens_to_add]).decode('UTF-8')}\x1b[0m \x1b[37m{sync.mhold.model.detokenize([int(a) for a in sync.dhold.best_beam_indices[1+sync.dhold.additional_sure_tokens:]]).decode('UTF-8')}\x1b[0m") # \[90m or \[37m for gray \x1b[43
            else:
                try:
                    print(f"current generation: {sync.mhold.tokenizer.decode(args['tokens'][0][sync.dhold.original_input_len:-len(sync.dhold.tokens_to_add)], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[32m{sync.mhold.tokenizer.decode([int(a) for a in sync.dhold.tokens_to_add], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m \x1b[37m{sync.mhold.tokenizer.decode([int(a) for a in sync.dhold.best_beam_indices[1+sync.dhold.additional_sure_tokens:]], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m")
                except:
                    print(f"current generation: {sync.mhold.processor.tokenizer.decode(args['tokens'].input_ids[0][sync.dhold.original_input_len:-len(sync.dhold.tokens_to_add)], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[32m{sync.mhold.processor.tokenizer.decode([int(a) for a in sync.dhold.tokens_to_add], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m \x1b[37m{sync.mhold.processor.tokenizer.decode([int(a) for a in sync.dhold.best_beam_indices[1+sync.dhold.additional_sure_tokens:]], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m")
            print("\n---------------------------------------------------------------\n\n")

    def beamsearch_get_returned_content(self, sync):
        args = sync.dhold.inputs
        if sync.dhold.was_list:
            sync.dhold.returned_content = [sync.mhold.model.detokenize(args['tokens'][sync.dhold.original_input_len:]).decode('UTF-8')]
        else:
            try:
                sync.dhold.returned_content = [sync.mhold.tokenizer.decode(args['tokens'].tolist()[0][sync.dhold.original_input_len:], skip_special_tokens=True)]
            except:
                sync.dhold.returned_content = [sync.mhold.processor.tokenizer.decode(args['tokens'].input_ids.tolist()[0][sync.dhold.original_input_len:], skip_special_tokens=True)]

    def beamsearch_check_break_condition(self, sync):
        sync.dhold.beamsearch_break = False
        args = sync.dhold.inputs
        if np.any(sync.dhold.tokens_to_add == sync.mhold.stop_token):
            if args['debugmode']:
                print("tokens to add contained stop token, stopping.")
            sync.dhold.beamsearch_break = True
        
        if sync.dhold.generated_tokens >= args['max_new_tokens']:
            sync.dhold.beamsearch_break = True
            if args['debugmode']:
                print("reached max_new_tokens, stopping.")

    def beamsearch_do_search(self, sync):
        args = sync.dhold.inputs
        if sync.dhold.considered_tokens_num[0] == 1:
            sync.dhold.tokens_to_add = [sync.dhold.logits[0, 0, 0, 0]]
            sync.dhold.best_path_indices = sync.dhold.tokens_to_add
            sync.dhold.additional_sure_tokens = 0
            sync.dhold.logits_merker = copy.deepcopy(sync.dhold.logits)
            sync.dhold.considered_tokens_num_merker = copy.deepcopy(sync.dhold.considered_tokens_num)
            sync.dhold.best_beam_indices = [sync.dhold.logits[0, 0, 0, 0]]
            
        else:
            sync.get_best_path()

            sync.dhold.tokens_to_add = [sync.dhold.best_beam_indices[0]] # at least at the init token for the best path
            sync.dhold.additional_sure_tokens = 0
            for i in range(1, len(sync.dhold.best_beam_indices)): # skip 0 since already added
                if sync.dhold.best_beam_probs[i] >= math.log(args['min_conf_for_sure']):
                    sync.dhold.additional_sure_tokens += 1
                    sync.dhold.tokens_to_add.append(sync.dhold.best_beam_indices[i])
                else:
                    break

    def inference_setup_args(self, sync):
        args = sync.dhold.inputs
        
        sync.mhold.stop_token = None
        sync.dhold.gen_kwargs = {
            'max_new_tokens': sync.dhold.max_tokens_this_gen,
            'do_sample': False,
            'temperature': 1,
            'output_scores': True,
            'return_dict_in_generate': True,
        }
        sync.dhold.got_input_shape_already = True
        

        
        if args['model'] in ["llama3-llava-next-8b", "Hermes-2-Theta-Llama-3-8B", "phi-3-vision-128k-instruct"]:
            sync.dhold.gen_function = sync.mhold.model.generate
            sync.dhold.get_logits = lambda output: find_top_indexes([token_logits.detach().cpu().numpy() for token_logits in output.scores], n_top=args['max_num_beams'])

        if args['model'] in ["llama3-llava-next-8b", "Hermes-2-Theta-Llama-3-8B"]:
            sync.dhold.original_input_len = args['tokens'].shape[-1]
            sync.dhold.attn_mask = torch.ones_like(args['tokens'], device=sync.config['torch_device'])
            sync.dhold.gen_kwargs.update({
                'num_beams': 1,
                'attention_mask': torch.ones_like(args['tokens'], device=sync.config['torch_device']),
                'pad_token_id': sync.mhold.tokenizer.eos_token_id,
            })
            sync.dhold.gen_input = args['tokens']
            sync.mhold.stop_token = sync.mhold.tokenizer.eos_token_id
        
        if args['model'] == "llama3-llava-next-8b":
            sync.dhold.gen_kwargs.update({
                'images': args['image_tensor'],
                'image_sizes': args['image_sizes'],
            })
            sync.dhold.output_processor = lambda output: [sync.mhold.tokenizer.decode(output.sequences[i][:], skip_special_tokens=True) for i in range(len(output.sequences))]
            sync.dhold.shape_attr = lambda output: output.sequences[0][:].shape

        
            
        if args['model'] == "Hermes-2-Theta-Llama-3-8B":
            sync.dhold.output_processor = lambda output: [sync.mhold.tokenizer.decode(output.sequences[i][sync.dhold.original_input_len:], skip_special_tokens=True) for i in range(len(output.sequences))]
            sync.dhold.shape_attr = lambda output: output.sequences[0][sync.dhold.original_input_len:].shape

        if args['model'] == "phi-3-vision-128k-instruct":
            sync.dhold.gen_kwargs.update({
                'attention_mask': torch.ones_like(args['tokens'].input_ids, device=sync.config['torch_device']),
                'pixel_values': args['tokens'].pixel_values if "pixel_values" in args['tokens'] else None,
                'image_sizes': args['tokens'].image_sizes if "image_sizes" in args['tokens'] else None,
                'eos_token_id': sync.mhold.processor.tokenizer.eos_token_id
            })
            sync.mhold.stop_token = sync.mhold.processor.tokenizer.eos_token_id
            sync.dhold.gen_input = args['tokens'].input_ids
            sync.dhold.output_processor = lambda output: [sync.mhold.processor.decode(output.sequences[i][args['tokens']['input_ids'].shape[1]:], skip_special_tokens=True) for i in range(len(output.sequences))]
            sync.dhold.shape_attr = lambda output: output.sequences[:, args['tokens']['input_ids'].shape[1]:].shape
            
        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
            del sync.dhold.gen_kwargs['max_new_tokens']
            del sync.dhold.gen_kwargs['do_sample']
            del sync.dhold.gen_kwargs['output_scores']
            del sync.dhold.gen_kwargs['return_dict_in_generate']
            sync.dhold.gen_kwargs.update({
                'max_tokens': sync.dhold.max_tokens_this_gen,
                'stop': ["<|eot_id|>"],
                'echo': False,
                'top_k': 1,
                'logprobs': -1,
            })
            sync.mhold.stop_token = sync.mhold.model.tokenize("<|eot_id|>".encode('UTF-8'), special=True)
            sync.dhold.got_input_shape_already = False
            sync.dhold.gen_function = sync.mhold.model
            sync.dhold.gen_input = args['tokens']
            sync.dhold.output_processor = lambda output: [out['text'] for out in output['choices']]
            sync.dhold.shape_attr = lambda output: [1, output['usage']['completion_tokens']]
            sync.dhold.input_shape_attr = lambda output: [1, output['usage']['prompt_tokens']]
            sync.dhold.get_logits = lambda scores: find_top_indexes(sync.mhold.model._scores[-sync.dhold.output_shape[-1]:], args['max_num_beams'])

    def inference_check_stop_token_and_alternative_inputs(self, sync):
        if sync.mhold.stop_token == None:
            raise Error('did/could not assign stop token')

        if sync.dhold.alternative_input != None:
            sync.dhold.gen_input = sync.dhold.alternative_input

        if sync.dhold.alternative_mask != None:
            sync.dhold.gen_kwargs.update({
                'attention_mask': sync.dhold.alternative_mask,
            })

    def inference_do_inference(self, sync):
        if not sync.dhold.llama_sequencial_batch:
            sync.dhold.gen_output = sync.dhold.gen_function(sync.dhold.gen_input, **sync.dhold.gen_kwargs)

            sync.dhold.returned_content = [entry.strip() for entry in sync.dhold.output_processor(sync.dhold.gen_output)]
            sync.dhold.output_shape = getattr(sync.dhold.gen_output, sync.dhold.shape_attr) if isinstance(sync.dhold.shape_attr, str) else sync.dhold.shape_attr(sync.dhold.gen_output)
            sync.dhold.logits = sync.dhold.get_logits(sync.dhold.gen_output)
        else:
            returned_content = []
            output_shape = []
            logits = []
            for entry in sync.dhold.gen_input:
                if isinstance(entry, torch.Tensor) and len(entry.shape) == 1:
                    entry = entry.unsqueeze(0)
                
                gen_output = sync.dhold.gen_function(sync.dhold.entry, **sync.dhold.gen_kwargs)
    
                returned_content.append([entry.strip() for entry in sync.dhold.output_processor(sync.dhold.gen_output)])
                output_shape.append(getattr(sync.dhold.gen_output, sync.dhold.shape_attr) if isinstance(sync.dhold.shape_attr, str) else sync.dhold.shape_attr(sync.dhold.gen_output))
                logits.append(sync.dhold.get_logits(sync.dhold.gen_output))
            sync.dhold.returned_content = returned_content
            sync.dhold.output_shape = np.array(output_shape)
            sync.dhold.logits = np.concatenate(logits, axis=0)

    def inference_get_considered_tokens_num(self, sync):
        args = sync.dhold.inputs
        # get number of considered tokens for each batch
        merker = [1 for _ in range(sync.dhold.logits.shape[0])] #  add the first one by default
        for batch_num in range(sync.dhold.logits.shape[0]):
            for top_logit_num in range(1, sync.dhold.logits.shape[2]):
                if sync.dhold.logits[batch_num][0][top_logit_num][1] >= args['min_conf_for_consider']:
                    merker[batch_num] += 1
                else: 
                    break
                if np.sum(sync.dhold.logits[batch_num][0][:merker[batch_num]][0]) >= args['prob_sum_for_search']:
                    break
        sync.dhold.considered_tokens_num = np.array(merker)

    def check_for_error_and_limit_tokens(self, sync):
        args = sync.dhold.gen_inputs
        if sync.dhold.error:
            return None
        if sync.dhold.limit_tokens != None:
            sync.dhold.max_tokens_this_gen = sync.dhold.limit_tokens
        else:
            sync.dhold.max_tokens_this_gen = args['max_new_tokens']

    def beamsearch_setup_inputs(self, sync):
        args = sync.dhold.inputs
        tokens = None
        try:
            tokens = args['tokens'].input_ids
        except:
            tokens = args['tokens']

        sync.dhold.was_list = False
        if isinstance(tokens, list):
            sync.dhold.was_list = True
            tokens = torch.tensor(tokens, device=sync.config['torch_device'])
            
        sync.dhold.batched_input_tokens = torch.concatenate((tokens.repeat(sync.dhold.considered_tokens_num[0], 1), torch.tensor(sync.dhold.logits[0, 0, :sync.dhold.considered_tokens_num[0], 0], device=sync.config['torch_device']).unsqueeze(1)), dim=-1).to(torch.long)
        sync.dhold.batched_input_masks = torch.ones_like(sync.dhold.batched_input_tokens, device=sync.config['torch_device'])

    def beamsearch_do_inference(self, sync):
        args = sync.dhold.inputs
        if sync.mhold.current_model == "llama3-llava-next-8b" and len(args['images']) > 0:
            sync.do_inference(limit_tokens=args['depth_beams'], alternative_input=sync.dhold.batched_input_tokens, alternative_mask=torch.ones_like(sync.dhold.batched_input_tokens[0].unsqueeze(0), device=sync.config['torch_device']), llama_sequencial_batch=True)
        elif sync.dhold.was_list:
            sync.do_inference(limit_tokens=args['depth_beams'], alternative_input=sync.dhold.batched_input_tokens.tolist(), llama_sequencial_batch=True)
        else:
            sync.do_inference(limit_tokens=args['depth_beams'], alternative_input=sync.dhold.batched_input_tokens, alternative_mask=sync.dhold.batched_input_masks)

    def beamsearch_get_beams_from_outputs(self, sync):
        for i in range(sync.dhold.considered_tokens_num[0]):
            # case considered token is stop token:
            if np.any(sync.dhold.logits_merker[0, 0, i, 0] == sync.mhold.stop_token):
                sync.dhold.total_probs.append(math.log(sync.dhold.logits_merker[0, 0, i, 1]))
                sync.dhold.prediction_paths_probs.append([math.log(sync.dhold.logits[0, 0, i, 1])])
                sync.dhold.prediction_paths_indices.append([sync.dhold.logits[0, 0, i, 0]])
                sync.dhold.skip_path.append(i)
                continue
                
            highest_path_probs = []
            highest_path_indices = []
            for token_num in range(sync.dhold.logits.shape[1]):
                highest_path_probs.append(math.log(sync.dhold.logits[i, token_num, 0, 1]))
                highest_path_indices.append(sync.dhold.logits[i, token_num, 0, 0])
                pass
            total_prob = math.log(sync.dhold.logits_merker[0, 0, i, 1])
            total_prob += sum(highest_path_probs)
            sync.dhold.total_probs.append(total_prob)
            sync.dhold.prediction_paths_probs.append([math.log(sync.dhold.logits_merker[0, 0, i, 1])]+highest_path_probs)
            sync.dhold.prediction_paths_indices.append([sync.dhold.logits_merker[0, 0, i, 0]]+highest_path_indices)

    def beamsearch_get_best_beam_from_beams(self, sync):
        if sync.dhold.inputs['debugmode']:
            print("paths total probs:", [round(entry, 3) for entry in sync.dhold.total_probs])

        best_beam = max(enumerate(sync.dhold.total_probs),key=lambda x: x[1])[0]

        sync.dhold.best_beam_probs = sync.dhold.prediction_paths_probs[best_beam]
        sync.dhold.best_beam_indices = sync.dhold.prediction_paths_indices[best_beam]