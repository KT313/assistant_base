from .imports import *
from .misc import softmax, find_top_indexes, functions, system_prompt

class ProcessorHelper():
    def __init__(self):
        pass

    def load_model(self, sync):
        if sync.mhold == None or sync.dhold.inputs['model'] != sync.mhold.current_model or sync.dhold.inputs['model_dtype'] != sync.mhold.current_dtype:
            sync.make_new_mhold()
            sync.mhold.load_model(sync, sync.dhold.inputs['model'], sync.dhold.inputs['model_dtype'])
        if sync.dhold.inputs['debugmode']: print(sync.dhold.inputs['model'], sync.mhold.current_model, flush=True)
    
    def build_prompt_string(self, sync):
        sync.dhold.inputs['chat'] = [chat for chat in sync.dhold.inputs['chat'] if chat['role'] != "System"]
        for i in range(len(sync.dhold.inputs['chat'])):
            old_role = sync.dhold.inputs['chat'][i]['role']
            if old_role == "User": sync.dhold.inputs['chat'][i]['role'] = "user"
            if old_role == "AI": sync.dhold.inputs['chat'][i]['role'] = "assistant"

        if sync.dhold.inputs['use_functions']:
            sync.dhold.inputs['chat'].insert(0, {'role': 'system', 'content': functions})
        elif 'manual_system_prompt' in sync.dhold.inputs and sync.dhold.inputs['manual_system_prompt'].strip() != "":
            sync.dhold.inputs['chat'].insert(0, {'role': 'system', 'content': sync.dhold.inputs['manual_system_prompt'].strip()})
        
        template_type = sync.config['models'][sync.dhold.inputs['model']]['template']
        template = sync.config['chat_templates'][template_type]

        prompt_string = ""
        prompt_string += template['init']

        if template['roles as string']:
            for index, entry in enumerate(sync.dhold.inputs['chat']):
                image_string = ""
                if index == (len(sync.dhold.inputs['chat'])-1) and len(sync.dhold.inputs['images']) > 0:
                    image_string = template['image token']
                prompt_string += f"{template['role start']}{entry['role']}{template['role end']}{image_string}{entry['content']}{template['end text']}"
            prompt_string += f"{template['role start']}assistant{template['role end']}"
        else:
            for index, entry in enumerate(sync.dhold.inputs['chat']):
                image_string = ""
                if index == (len(sync.dhold.inputs['chat'])-1) and len(sync.dhold.inputs['images']) > 0:
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
        sync.dhold.inputs['max_num_beams'] = int(sync.dhold.inputs['beam_config']['max_num_beams'].strip())
        sync.dhold.inputs['depth_beams'] = int(sync.dhold.inputs['beam_config']['depth_beams'].strip())
        sync.dhold.inputs['min_conf_for_sure'] = float(sync.dhold.inputs['beam_config']['min_conf_for_sure'].strip())
        sync.dhold.inputs['min_conf_for_consider'] = float(sync.dhold.inputs['beam_config']['min_conf_for_consider'].strip())
        sync.dhold.inputs['prob_sum_for_search'] = float(sync.dhold.inputs['beam_config']['prob_sum_for_search'].strip())
    def prepare_model_generation_args(self, sync):
        image_tensor = None
        img_token_index = None
        image_sizes = None
        image_input = None

        if sync.mhold.image_capable:
            if len(sync.dhold.inputs['images']) > 0:
                # for llama3-llava
                if sync.mhold.current_model in ["llama3-llava-next-8b"]:
                    image_tensor = process_images(sync.dhold.inputs['images'], sync.mhold.image_processor, sync.mhold.model.config)
                    image_tensor = [_image.to(dtype=torch.float16, device=sync.config['torch_device']) for _image in image_tensor]
                    image_sizes = [image.size for image in sync.dhold.inputs['images']]
                    img_token_index = IMAGE_TOKEN_INDEX

                # for phi3-vision
                elif sync.mhold.current_model in ["phi-3-vision-128k-instruct"]:
                    image_input = sync.dhold.inputs['images']

        # vision models separately
        if sync.mhold.current_model in ["llama3-llava-next-8b"]:
            tokenizer_output = tokenizer_image_token(sync.dhold.prompt_string, sync.mhold.tokenizer, img_token_index, return_tensors="pt").unsqueeze(0).to(sync.config['torch_device'])
            sync.dhold.inputs['tokens'] = tokenizer_output
            sync.dhold.inputs['image_tensor'] = image_tensor
            sync.dhold.inputs['image_sizes'] = [image.size for image in sync.dhold.inputs['images']]
            sync.dhold.input_shape = sync.dhold.inputs['tokens'].shape #, sync.dhold.inputs['image_tensor'][0].shape] TODO: find out image token size
        elif sync.mhold.current_model in ["phi-3-vision-128k-instruct"]:
            tokenizer_output = sync.mhold.processor(sync.dhold.prompt_string, image_input, return_tensors="pt").to(sync.config['torch_device'])
            sync.dhold.inputs['tokens'] = tokenizer_output.input_ids
            sync.dhold.inputs['pixel_values'] = tokenizer_output.pixel_values if "pixel_values" in tokenizer_output else None
            sync.dhold.inputs['image_sizes'] = tokenizer_output.image_sizes if "image_sizes" in tokenizer_output else None
            sync.dhold.input_shape = sync.dhold.inputs['tokens'].shape

        # then general backend types
        elif sync.mhold.backend == "transformers":
            tokenizer_output = sync.mhold.tokenizer(sync.dhold.prompt_string, return_tensors="pt").input_ids.to(sync.config['torch_device'])
            
            sync.dhold.inputs['tokens'] = tokenizer_output
            sync.dhold.input_shape = sync.dhold.inputs['tokens'].shape
        elif sync.mhold.backend == "llama-cpp":
            sync.dhold.inputs['tokens'] = torch.tensor(sync.mhold.model.tokenize(sync.dhold.prompt_string.encode('UTF-8')), device=sync.config['torch_device'])
            sync.dhold.input_shape = [1, len(sync.dhold.inputs['tokens'])]

        sync.dhold.original_input_len = sync.dhold.inputs['tokens'].shape[-1]

        

    def append_tokens_to_add_to_tokens(self, sync):
        tokens = sync.dhold.inputs['tokens']
            
        if sync.mhold.backend == "llama-cpp":
            if not torch.is_tensor(tokens):
                tokens = torch.tensor(tokens, device=sync.config['torch_device'])
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)

        sync.dhold.inputs['tokens'] = torch.concatenate((tokens, torch.tensor(sync.dhold.tokens_to_add, device=sync.config['torch_device']).to(torch.int).unsqueeze(0)), dim=-1)


    
    def print_beam_debug_info(self, sync):
        if sync.dhold.inputs['debugmode']:
            print(" | ".join([str(round(entry, 5)).ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 1]]))
            print(" | ".join([entry.strip().ljust(14) for entry in sync.mhold.detokenize(sync.dhold.logits_merker[0, 0, :, 0].astype(np.int32), split=True)]))
            if np.any(sync.dhold.considered_tokens_num_merker == 1): print("-> single considered token, not doing beam search")
            else: print(f"-> using {sync.dhold.considered_tokens_num_merker} beams")
                

            print("\n")
            print(f"current generation: {''.join(sync.mhold.detokenize(sync.dhold.inputs['tokens'][0][sync.dhold.original_input_len:-len(sync.dhold.tokens_to_add)]))}\x1b[32m{''.join(sync.mhold.detokenize(sync.dhold.tokens_to_add))}\x1b[0m \x1b[37m{''.join(sync.mhold.detokenize(sync.dhold.best_beam_indices[1+sync.dhold.additional_sure_tokens:]))}\x1b[0m") # \[90m or \[37m for gray \x1b[43
            print("\n---------------------------------------------------------------\n\n")

    

    def inference_setup_args(self, sync):        
        sync.mhold.stop_token = None
        if sync.dhold.inputs['beam_config']['use_beam_search']:
            sync.dhold.gen_kwargs = {
                'max_new_tokens': sync.dhold.max_tokens_this_gen,
                'do_sample': False,
                'temperature': 1,
                'output_scores': True,
                'return_dict_in_generate': True,
            }
        else:
            sync.dhold.gen_kwargs = {
                'max_new_tokens': sync.dhold.max_tokens_this_gen,
                'do_sample': False,
                'temperature': 1,
                # 'output_scores': True,
                'return_dict_in_generate': True,
            }
        sync.dhold.got_input_shape_already = True
        

        
        if sync.dhold.inputs['model'] in ["llama3-llava-next-8b", "Hermes-2-Theta-Llama-3-8B", "phi-3-vision-128k-instruct"]:
            sync.dhold.gen_function = sync.mhold.model.generate
            sync.dhold.get_logits = lambda output: find_top_indexes(sync, [token_logits.detach().cpu().numpy() for token_logits in output.scores], n_top=sync.dhold.inputs['max_num_beams'])

        if sync.dhold.inputs['model'] in ["llama3-llava-next-8b", "Hermes-2-Theta-Llama-3-8B"]:
            sync.dhold.gen_kwargs.update({
                'num_beams': 1,
                'pad_token_id': sync.mhold.tokenizer.eos_token_id,
            })
            sync.dhold.gen_input = sync.dhold.inputs['tokens'].to(torch.int32)
            sync.mhold.stop_token = sync.mhold.tokenizer.eos_token_id
        
        if sync.dhold.inputs['model'] == "llama3-llava-next-8b":
            sync.dhold.gen_kwargs.update({
                'images': sync.dhold.inputs['image_tensor'],
                'image_sizes': sync.dhold.inputs['image_sizes'],
            })
            sync.dhold.output_processor = lambda output: [sync.mhold.tokenizer.decode(output.sequences[i][:], skip_special_tokens=True) for i in range(len(output.sequences))]
            sync.dhold.shape_attr = lambda output: output.sequences[0][:].shape

        
            
        if sync.dhold.inputs['model'] == "Hermes-2-Theta-Llama-3-8B":
            sync.dhold.output_processor = lambda output: [sync.mhold.tokenizer.decode(output.sequences[i][sync.dhold.original_input_len:], skip_special_tokens=True) for i in range(len(output.sequences))]
            sync.dhold.shape_attr = lambda output: output.sequences[0][sync.dhold.original_input_len:].shape

        if sync.dhold.inputs['model'] == "phi-3-vision-128k-instruct":
            sync.dhold.gen_kwargs.update({
                'pixel_values': sync.dhold.inputs['pixel_values'],
                'image_sizes': sync.dhold.inputs['image_sizes'],
                'eos_token_id': sync.mhold.tokenizer.eos_token_id
            })
            sync.mhold.stop_token = sync.mhold.tokenizer.eos_token_id
            sync.dhold.gen_input = sync.dhold.inputs['tokens'].to(torch.int32)
            sync.dhold.output_processor = lambda output: [sync.mhold.processor.decode(output.sequences[i][sync.dhold.inputs['tokens'].shape[1]:], skip_special_tokens=True) for i in range(len(output.sequences))]
            sync.dhold.shape_attr = lambda output: output.sequences[:, sync.dhold.inputs['tokens'].shape[1]:].shape
            
        if sync.dhold.inputs['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or sync.dhold.inputs['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
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
            sync.dhold.gen_function = lambda gen_input, **gen_kwargs: sync.mhold.model(gen_input.tolist() if torch.is_tensor(gen_input) else gen_input, **gen_kwargs)

            sync.dhold.gen_input = sync.dhold.inputs['tokens'].to(torch.int32)
            sync.dhold.output_processor = lambda output: [out['text'] for out in output['choices']]
            sync.dhold.shape_attr = lambda output: [1, output['usage']['completion_tokens']] if output['choices'][0]['finish_reason'] != "stop" else [1, output['usage']['completion_tokens']+1] # min shape 1 because stop token output results in completion tokens 0 for some reason
            sync.dhold.input_shape_attr = lambda output: [1, output['usage']['prompt_tokens']]
            sync.dhold.get_logits = lambda scores: find_top_indexes(sync, sync.mhold.model._scores[-sync.dhold.output_shape[-1]:], sync.dhold.inputs['max_num_beams'])

        if sync.mhold.backend == "transformers":
            sync.mhold.stop_token = [sync.mhold.stop_token]
            sync.mhold.stop_token.append(sync.mhold.tokenizer("<|eot_id|>").input_ids[0])
    
    def inference_check_stop_token_and_alternative_inputs(self, sync):
        if sync.mhold.stop_token == None:
            raise Error('did/could not assign stop token')

        if sync.dhold.alternative_input != None:
            sync.dhold.gen_input = sync.dhold.alternative_input

    def inference_do_inference(self, sync):
        with torch.no_grad():
            try:
                if sync.mhold.backend == "llama-cpp":
                    # make 2d list to 1d list if not batch for beam search
                    if not sync.dhold.sequencial_batch and sync.mhold.backend == "llama-cpp" and len(sync.dhold.gen_input.shape) > 1:
                        sync.dhold.gen_input = sync.dhold.gen_input[0]

                if sync.dhold.inputs['debugmode']: print(f"gen_input shape: {sync.dhold.gen_input.shape}, type: {sync.dhold.gen_input.dtype}")
                if not sync.dhold.sequencial_batch:
                    if sync.dhold.inputs['debugmode']: print("inference will start (not sequential batch)")
                    sync.dhold.stopped = [False for _ in range(sync.dhold.inputs['max_num_beams'])]
                    if not sync.mhold.backend == "llama-cpp":
                        sync.dhold.gen_kwargs['attention_mask'] = torch.ones_like(sync.dhold.gen_input, device=sync.config['torch_device'])
                    sync.dhold.gen_output = sync.dhold.gen_function(sync.dhold.gen_input, **sync.dhold.gen_kwargs)
        
                    sync.dhold.returned_content = [entry for entry in sync.dhold.output_processor(sync.dhold.gen_output)]
                    sync.dhold.output_shape = getattr(sync.dhold.gen_output, sync.dhold.shape_attr) if isinstance(sync.dhold.shape_attr, str) else sync.dhold.shape_attr(sync.dhold.gen_output)
                    if sync.dhold.inputs['beam_config']['use_beam_search']:
                        sync.dhold.logits = sync.dhold.get_logits(sync.dhold.gen_output)
                else:
                    if sync.dhold.inputs['debugmode']: print("inference will start (sequential batch)")
                    returned_content = []
                    output_shape = []
                    logits = []
                    stopped = []
                    sync.dhold.gen_kwargs['stop'] = []
                    for index, entry in enumerate(sync.dhold.gen_input):
                        if isinstance(entry, torch.Tensor) and len(entry.shape) == 1:
                            entry = entry.unsqueeze(0)
                        if not sync.mhold.backend == "llama-cpp":
                            sync.dhold.gen_kwargs['attention_mask'] = torch.ones_like(entry, device=sync.config['torch_device'])
                        sync.dhold.gen_output = sync.dhold.gen_function(entry, **sync.dhold.gen_kwargs)
                        if sync.dhold.gen_output['choices'][0]['finish_reason'] == "stop":
                            stopped.append(True)
                        else:
                            stopped.append(False)
                        # if sync.dhold.gen_output['choices'][0]['finish_reason'] == "stop":
                        #     sync.dhold.gen_output[choices] = [choice[text] for choice in sync.dhold.gen_output[choices]]
                        
                        returned_content.append([entry for entry in sync.dhold.output_processor(sync.dhold.gen_output)])
        
                        merker = getattr(sync.dhold.gen_output, sync.dhold.shape_attr) if isinstance(sync.dhold.shape_attr, str) else sync.dhold.shape_attr(sync.dhold.gen_output)
                        merker[0] = len(sync.dhold.gen_input)
                        sync.dhold.output_shape = merker
                        logits.append(sync.dhold.get_logits(sync.dhold.gen_output))
        
                    sync.dhold.returned_content = returned_content
                    sync.dhold.output_shape = np.array(output_shape)
                    if sync.dhold.inputs['beam_config']['use_beam_search']:
                        sync.dhold.logits = np.concatenate(logits, axis=0)
                    sync.dhold.stopped = stopped
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Caught OOM exception:", e)
                raise e
            if sync.dhold.inputs['debugmode']: print("inference ended gracefully")
        
            
    def inference_get_considered_tokens_num(self, sync):
        # get number of considered tokens for each batch
        merker = [1 for _ in range(sync.dhold.logits.shape[0])] #  add the first one by default
        for batch_num in range(sync.dhold.logits.shape[0]):
            for top_logit_num in range(1, sync.dhold.logits.shape[2]):
                if sync.dhold.logits[batch_num][0][top_logit_num][1] >= sync.dhold.inputs['min_conf_for_consider']:
                    merker[batch_num] += 1
                else: 
                    break
                if np.sum(sync.dhold.logits[batch_num][0][:merker[batch_num]][0]) >= sync.dhold.inputs['prob_sum_for_search']:
                    break
        sync.dhold.considered_tokens_num = np.array(merker)
    def inference_check_for_error_and_limit_tokens(self, sync):
        if sync.dhold.error:
            return None
        if sync.dhold.limit_tokens != None:
            sync.dhold.max_tokens_this_gen = sync.dhold.limit_tokens
        else:
            sync.dhold.max_tokens_this_gen = sync.dhold.inputs['max_new_tokens']

    def beamsearch_setup_inputs(self, sync):
        sync.dhold.total_probs  = []
        sync.dhold.prediction_paths_probs = []
        sync.dhold.prediction_paths_indices = []
        sync.dhold.skip_path = []

        sync.dhold.logits_merker = copy.deepcopy(sync.dhold.logits)
        sync.dhold.considered_tokens_num_merker = copy.deepcopy(sync.dhold.considered_tokens_num)
        
        tokens = sync.dhold.inputs['tokens']
            
        sync.dhold.batched_input_tokens = torch.concatenate((tokens.repeat(sync.dhold.considered_tokens_num[0], 1), torch.tensor(sync.dhold.logits[0, 0, :sync.dhold.considered_tokens_num[0], 0], device=sync.config['torch_device']).unsqueeze(1)), dim=-1).to(torch.int)
    def beamsearch_do_inference(self, sync):
        if sync.mhold.current_model == "llama3-llava-next-8b" and len(sync.dhold.inputs['images']) > 0:
            sync.do_inference(limit_tokens=sync.dhold.inputs['depth_beams'], alternative_input=sync.dhold.batched_input_tokens, sequencial_batch=True)
        elif sync.mhold.current_model in ["Meta-Llama-3-70B-Instruct-IQ2_S", "Meta-Llama-3-70B-Instruct-IQ1_M"]:
            sync.do_inference(limit_tokens=sync.dhold.inputs['depth_beams'], alternative_input=sync.dhold.batched_input_tokens.tolist(), sequencial_batch=True)
        else:
            sync.do_inference(limit_tokens=sync.dhold.inputs['depth_beams'], alternative_input=sync.dhold.batched_input_tokens)
    def beamsearch_get_beams_from_outputs(self, sync):
        for i in range(sync.dhold.considered_tokens_num_merker[0]):
            # case considered token is stop token:
            if np.any(sync.dhold.logits_merker[0, 0, i, 0] == sync.mhold.stop_token):
                sync.dhold.total_probs.append(math.log(sync.dhold.logits_merker[0, 0, i, 1]))
                sync.dhold.prediction_paths_probs.append([math.log(sync.dhold.logits[0, 0, i, 1])])
                sync.dhold.prediction_paths_indices.append([sync.dhold.logits[0, 0, i, 0].astype(np.int32)])
                sync.dhold.skip_path.append(i)
                continue
                
            highest_path_probs = []
            highest_path_indices = []
            for token_num in range(sync.dhold.logits.shape[1]):
                highest_path_probs.append(math.log(sync.dhold.logits[i, token_num, 0, 1]))
                highest_path_indices.append(sync.dhold.logits[i, token_num, 0, 0].astype(np.int32))
                pass
            total_prob = math.log(sync.dhold.logits_merker[0, 0, i, 1])
            total_prob += sum(highest_path_probs)
            sync.dhold.total_probs.append(total_prob)
            sync.dhold.prediction_paths_probs.append([math.log(sync.dhold.logits_merker[0, 0, i, 1])]+highest_path_probs)
            sync.dhold.prediction_paths_indices.append([sync.dhold.logits_merker[0, 0, i, 0].astype(np.int32)]+highest_path_indices)
    def beamsearch_get_best_beam_from_beams(self, sync):
        if sync.dhold.inputs['debugmode']:
            print("paths total probs:", [round(entry, 3) for entry in sync.dhold.total_probs])

        best_beam = max(enumerate(sync.dhold.total_probs),key=lambda x: x[1])[0]
        sync.dhold.best_beam_index = best_beam

        sync.dhold.best_beam_probs = sync.dhold.prediction_paths_probs[best_beam]
        sync.dhold.best_beam_indices = sync.dhold.prediction_paths_indices[best_beam]
    def beamsearch_get_returned_content(self, sync):
        sync.dhold.returned_content = sync.mhold.detokenize(sync.dhold.inputs['tokens'][0][sync.dhold.original_input_len:], skip_special=True)
    def beamsearch_check_break_condition(self, sync):
        sync.dhold.beamsearch_break = False
        if "best_beam_index" in sync.dhold.__dict__ and sync.dhold.stopped[sync.dhold.best_beam_index]:
            was_stopped = True
        else:
            was_stopped= False
        if np.any(np.isin(sync.dhold.tokens_to_add, sync.mhold.stop_token)) or was_stopped:
            if sync.dhold.inputs['debugmode']:
                print("tokens to add contained stop token, stopping.")
            sync.dhold.beamsearch_break = True
        
        if sync.dhold.generated_tokens >= sync.dhold.inputs['max_new_tokens']:
            sync.dhold.beamsearch_break = True
            if sync.dhold.inputs['debugmode']:
                print("reached max_new_tokens, stopping.")
    def beamsearch_do_search(self, sync):
        sync.dhold.considered_tokens_num_merker = copy.deepcopy(sync.dhold.considered_tokens_num)
        if sync.dhold.considered_tokens_num_merker[0] == 1:
            sync.dhold.tokens_to_add = [sync.dhold.logits[0, 0, 0, 0].astype(np.int32)]
            sync.dhold.best_path_indices = sync.dhold.tokens_to_add
            sync.dhold.additional_sure_tokens = 0
            sync.dhold.logits_merker = copy.deepcopy(sync.dhold.logits)
            sync.dhold.best_beam_indices = [sync.dhold.logits_merker[0, 0, 0, 0].astype(np.int32)]
            
        else:
            sync.get_best_path()

            sync.dhold.tokens_to_add = [sync.dhold.best_beam_indices[0]] # at least at the init token for the best path
            sync.dhold.additional_sure_tokens = 0
            for i in range(1, len(sync.dhold.best_beam_indices)): # skip 0 since already added
                if sync.dhold.best_beam_probs[i] >= math.log(sync.dhold.inputs['min_conf_for_sure']):
                    sync.dhold.additional_sure_tokens += 1
                    sync.dhold.tokens_to_add.append(sync.dhold.best_beam_indices[i])
                else:
                    break