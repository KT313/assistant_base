from .imports import *
from .misc import softmax, find_top_indexes, functions, system_prompt

class ProcessorHelper():
    def __init__(self):
        pass

    def load_model(self, sync):
        """
        checks if a model has to be loaded by comparing 
        currently loaded model info from sync.mhold to 
        requested model in sync.dhold.inputs
        """
        
        if sync.mhold == None or sync.dhold.inputs['model'] != sync.mhold.current_model or sync.dhold.inputs['model_dtype'] != sync.mhold.current_dtype:
            sync.make_new_mhold()
            sync.mhold.load_model(sync, sync.dhold.inputs['model'], sync.dhold.inputs['model_dtype'])
        if sync.dhold.inputs['debugmode']: print(sync.dhold.inputs['model'], sync.mhold.current_model, flush=True)
    
    def build_prompt_string(self, sync):
        """
        builds prompt string based on info in config file 
        and sync.dhold.inputs and stores it in 
        sync.dhold.prompt_string
        """
        
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

        user_role = template['user role'] if "user role" in template else "user"
        ai_role = template['ai role'] if "ai role" in template else "assistant"

        def convert_role(role):
            if role == "user":
                new_role_name = user_role
            elif role == "assistant":
                new_role_name = ai_role
            else:
                new_role_name = role
            return new_role_name

        index_last_user_msg = 0
        index_last_assistant_msg = 0
        for i in range(len(sync.dhold.inputs['chat'])-1, -1, -1):
            if sync.dhold.inputs['chat'][i]['role'] == "user":
                index_last_user_msg = i
                break
        for i in range(len(sync.dhold.inputs['chat'])-1, -1, -1):
            if sync.dhold.inputs['chat'][i]['role'] == "assistant":
                index_last_assistant_msg = i
                break
            

        if template['roles as string']:
            for index, entry in enumerate(sync.dhold.inputs['chat']):
                image_string = ""
                if index == index_last_user_msg and len(sync.dhold.inputs['images']) > 0:
                    image_string = template['image token']
                if index == (len(sync.dhold.inputs['chat'])-1) and index == index_last_assistant_msg:
                    prompt_string += f"{template['role start']}{convert_role(entry['role'])}{template['role end']}{entry['content']}"
                else:
                    prompt_string += f"{template['role start']}{convert_role(entry['role'])}{template['role end']}{image_string}{entry['content']}{template['end text']}"
            if index_last_user_msg > index_last_assistant_msg:
                prompt_string += f"{template['role start']}assistant{template['role end']}"
        else:
            for index, entry in enumerate(sync.dhold.inputs['chat']):
                image_string = ""
                if index == index_last_user_msg and len(sync.dhold.inputs['images']) > 0:
                    image_string = template['image token']
                if entry['role'] == "system":
                    role_token = template['system role']
                elif entry['role'] == "user":
                    role_token = template['user role']
                elif entry['role'] == "assistant":
                    role_token = template['ai role']
                if index == (len(sync.dhold.inputs['chat'])-1) and index == index_last_assistant_msg:
                    prompt_string += f"{role_token}{entry['content']}"
                else:
                    prompt_string += f"{role_token}{image_string}{entry['content']}{template['end text']}"
            if index_last_user_msg > index_last_assistant_msg:
                prompt_string += f"{template['ai role']}"

        sync.dhold.prompt_string = prompt_string
        if sync.dhold.inputs['debugmode']: print(f"built prompt string:\n\"{sync.dhold.prompt_string}\"")
    def load_beam_config(self, sync):   
        """
        converts beam config values in sync.dhold.inputs 
        to the appropriate format
        """
        
        sync.dhold.inputs['max_num_beams'] = int(sync.dhold.inputs['beam_config']['max_num_beams'].strip())
        sync.dhold.inputs['depth_beams'] = int(sync.dhold.inputs['beam_config']['depth_beams'].strip())
        sync.dhold.inputs['min_conf_for_sure'] = float(sync.dhold.inputs['beam_config']['min_conf_for_sure'].strip())
        sync.dhold.inputs['min_conf_for_consider'] = float(sync.dhold.inputs['beam_config']['min_conf_for_consider'].strip())
        sync.dhold.inputs['prob_sum_for_search'] = float(sync.dhold.inputs['beam_config']['prob_sum_for_search'].strip())
    def prepare_model_generation_args(self, sync):
        """
        takes inputs from sync.dhold.inputs to convert them 
        to right formats and save them to sync.dhold
        """

        
        tokenizer_output = sync.mhold.helper.encode(sync.dhold.prompt_string, images=sync.dhold.inputs['images'])
        print("tokenizer out ids:", tokenizer_output['ids'])
        del sync.dhold.inputs['images']
        for key, val in tokenizer_output.additional_data.items():
            sync.dhold.inputs[key] = val
        
        sync.dhold.inputs['tokens'] = tokenizer_output['ids'].to(sync.config['torch_device'])
        sync.dhold.input_shape = sync.dhold.inputs['tokens'].shape
        if "position_offsets" in tokenizer_output:
            sync.dhold.inputs['position_offsets'] = tokenizer_output['position_offsets'].to(sync.config['torch_device'])

        sync.dhold.original_input_len = sync.dhold.inputs['tokens'].shape[-1]

        if sync.dhold.inputs['debugmode']: 
            print(f"stop token: {sync.mhold.helper.stop_token}, decoded: {sync.mhold.helper.decode(sync.mhold.helper.stop_token, skip_special_tokens=False)}")

        

    def append_tokens_to_add_to_tokens(self, sync):
        """
        takes tokens from sync.dhold.tokens_to_add and adds 
        them to sync.dhold.inputs['tokens']
        """
        if sync.dhold.inputs['debugmode']: print(f"untrimmed before appending: {sync.mhold.helper.decode(sync.dhold.inputs['tokens'], skip_special_tokens=False)}")
        if sync.dhold.inputs['debugmode']: print(f"will append these tensors: {sync.mhold.helper.decode(sync.dhold.inputs['tokens'][0:1, :-1], skip_special_tokens=False)}, {sync.mhold.helper.decode(torch.tensor(sync.dhold.tokens_to_add, device=sync.config['torch_device']).to(torch.int).unsqueeze(0), skip_special_tokens=False)}")

        if sync.dhold.inputs['tokens'].shape[0] > 1:
            tokens = sync.dhold.inputs['tokens'][0:1, :-1] # remove the temp tokens for batch beam search
        else:
            # do not strip last token if no beamsearch was done (no batched inputs)
            tokens = sync.dhold.inputs['tokens'][0:1, :]
        sync.dhold.inputs['tokens'] = torch.concatenate([tokens, torch.tensor(sync.dhold.tokens_to_add, device=sync.config['torch_device']).to(torch.int).unsqueeze(0)], dim=-1)

        if sync.dhold.inputs['debugmode']: print(f"append result: {sync.mhold.helper.decode(sync.dhold.inputs['tokens'], skip_special_tokens=False)}")


    
    def print_beam_debug_info(self, sync):
        """
        prints debug info about beam search
        """
        
        if sync.dhold.inputs['debugmode']:
            
            print(" | ".join([str(round(entry.item(), 5)).ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 1]]))
            print(" | ".join([entry.strip().ljust(14) for entry in sync.mhold.helper.decode(sync.dhold.logits_merker[0, 0, :, 0].to(torch.int32), logits_mode=True)]))
            print(" | ".join([str(entry.item()).strip().ljust(14) for entry in sync.dhold.logits_merker[0, 0, :, 0].to(torch.int32)]))
            if torch.any(torch.tensor(sync.dhold.considered_tokens_num_merker == 1)): print("-> single considered token, not doing beam search")
            else: print(f"-> using {sync.dhold.considered_tokens_num_merker} beams")
                

            print("\n")
            print(f"current generation: {''.join(sync.mhold.helper.decode(sync.dhold.inputs['tokens'][0][sync.dhold.original_input_len:-len(sync.dhold.tokens_to_add)], skip_special_tokens=False, logits_mode=False)[0])}\x1b[32m{''.join(sync.mhold.helper.decode(sync.dhold.tokens_to_add, skip_special_tokens=False, logits_mode=False)[0])}\x1b[0m \x1b[37m{''.join(sync.mhold.helper.decode(sync.dhold.best_beam_indices[1+sync.dhold.additional_sure_tokens:], skip_special_tokens=False, logits_mode=False)[0])}\x1b[0m") # \[90m or \[37m for gray \x1b[43
        print("\n---------------------------------------------------------------\n\n")

    

    def inference_setup_args(self, sync):  
        """
        generates the generation arguments dict based on 
        currently loaded model
        """
        
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
                'eos_token_id': sync.mhold.helper.stop_token,
                # 'stop_token': sync.mhold.tokenizer.eos_token_id
                #"eos_token_id": [128001, 128009],
            }
        

        if sync.dhold.inputs['model'] in ["test"]:
            sync.dhold.gen_kwargs.update({
                'position_offsets': sync.dhold.inputs['position_offsets']
            })


        for key in ["pixel_values", "image_sizes", "images"]:
            try:
                sync.dhold.gen_kwargs.update({
                    key: sync.dhold.inputs[key]
                })
            except:
                pass






            
        if sync.dhold.inputs['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or sync.dhold.inputs['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
            for key in ["max_new_tokens", "do_sample", "output_scores", "return_dict_in_generate", "eos_token_id"]:
                try:
                    del sync.dhold.gen_kwargs[key]
                except:
                    if sync.dhold.inputs['debugmode']: print(f"could not remove '{key}' from sync.dhold.gen_kwargs, maybe did not exist")
                    pass

            sync.dhold.gen_kwargs.update({
                'max_tokens': sync.dhold.max_tokens_this_gen,
                'stop': ["<|eot_id|>"],
                'echo': False,
                'top_k': 1,
                'logprobs': -1,
            })


    
    
    def inference_check_stop_token_and_alternative_inputs(self, sync):
        """
        checks if a stop token is assigned, if yes: raise Error
        checks if an alternative input is provided, if yes: overwrite sync.dhold.gen_input with it
        """
        
        if sync.mhold.helper.stop_token == None:
            raise Exception('did/could not assign stop token')

        if sync.dhold.alternative_input != None:
            sync.dhold.gen_input = sync.dhold.alternative_input

    def inference_do_inference(self, sync):
        """
        main generation part, different for each model and 
        sequencial_batch mode. 
        stores outputs in sync.dhold.returned_content. 
        also checks if llama-cpp inference was stopped 
        because of stop token
        """
        
        with torch.no_grad():
            try:
                generation_dict = sync.mhold.helper.generate(sync.dhold.inputs['tokens'].to(torch.int32), **sync.dhold.gen_kwargs)
                if sync.dhold.inputs['debugmode']:
                    for key, val in generation_dict.items():
                        print(f"{key}: {type(val)}")
    
                sync.dhold.returned_content = generation_dict['decoded_output']
                sync.dhold.output_shape = generation_dict['output_shape']
                sync.dhold.logits = generation_dict['top_logits']
    
                sync.dhold.stopped = [False for _ in range(sync.dhold.inputs['max_num_beams'])]
    
                if sync.dhold.inputs['beam_config']['use_beam_search']:
                    if "stopped" in generation_dict:
                        sync.dhold.stopped = generation_dict['stopped']
        
                    if sync.dhold.logits.shape[0] == 1 and sync.dhold.stopped[0]:
                        sync.dhold.beamsearch_break = True
                        print("beam search stopped since only beam contains stop")

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print('CUDA out of memory error caught!', e)
                    sync.dhold.error = True
                    sync.dhold.error_info = "CUDA out of memory"
                else:
                    print(e)
                    sync.dhold.error = True
                    sync.dhold.error_info = str(e)
    
            
    def inference_get_considered_tokens_num(self, sync):
        """
        looks at sync.dhold.logits and adds the number of 
        logits with a probability of at least 
        min_conf_for_consider to 
        sync.dhold.considered_tokens_num
        """
        
        # get number of considered tokens for each batch
        merker = [1 for _ in range(sync.dhold.logits.shape[0])] #  add the first one by default
        for batch_num in range(sync.dhold.logits.shape[0]):
            for top_logit_num in range(1, sync.dhold.logits.shape[2]):
                if sync.dhold.logits[batch_num][0][top_logit_num][1] >= sync.dhold.inputs['min_conf_for_consider']:
                    merker[batch_num] += 1
                else: 
                    break
                if torch.sum(sync.dhold.logits[batch_num][0][:merker[batch_num]][0]) >= sync.dhold.inputs['prob_sum_for_search']:
                    break
        sync.dhold.considered_tokens_num = torch.tensor(merker)
    def inference_check_for_error_and_limit_tokens(self, sync):
        """
        checks if there was an error by looking at sync.dhold.error
        sets token generation limit to sync.dhold.limit_tokens if it is not None
        """
        
        if sync.dhold.error:
            return None
        if sync.dhold.limit_tokens != None:
            sync.dhold.max_tokens_this_gen = sync.dhold.limit_tokens
        else:
            sync.dhold.max_tokens_this_gen = sync.dhold.inputs['max_new_tokens']

    def beamsearch_setup_inputs(self, sync):
        """
        saves old logits and number of considered tokens to 
        a merker since the next generation would overwrite 
        them
        creates a batch of inputs for the next beam search 
        where everything except the last token is the same
        """
        
        sync.dhold.total_probs  = []
        sync.dhold.prediction_paths_probs = []
        sync.dhold.prediction_paths_indices = []
        sync.dhold.skip_path = []

        sync.dhold.logits_merker = copy.deepcopy(sync.dhold.logits)
        sync.dhold.considered_tokens_num_merker = copy.deepcopy(sync.dhold.considered_tokens_num)
        
        tokens = sync.dhold.inputs['tokens']

        if sync.dhold.inputs['debugmode']: print("before batching:", sync.mhold.helper.decode(sync.dhold.inputs['tokens']))
        sync.dhold.batched_input_tokens = torch.concatenate((tokens.repeat(sync.dhold.considered_tokens_num[0], 1), torch.tensor(sync.dhold.logits[0, 0, :sync.dhold.considered_tokens_num[0], 0], device=sync.config['torch_device']).unsqueeze(1)), dim=-1).to(torch.int)
        sync.dhold.inputs['tokens'] = sync.dhold.batched_input_tokens
        if sync.dhold.inputs['debugmode']: print("after batching:", sync.mhold.helper.decode(sync.dhold.inputs['tokens']))
    def beamsearch_do_inference(self, sync):
        """
        starts sync.do_inference with limit_tokens set to 
        beam depth
        depends on model since llama3-llava with image and 
        llama-cpp models cannot do batch inference
        """
        
        sync.do_inference(limit_tokens=sync.dhold.inputs['depth_beams'], alternative_input=sync.dhold.batched_input_tokens)
        
    def beamsearch_get_beams_from_outputs(self, sync):
        """
        appends probabilities and indexes from generation 
        outputs to the 
        sync.dhold.prediction_paths_probs/indices lists 
        created in beamsearch_setup_inputs
        
        appends total probability of each beam to 
        sync.dhold.total_probs
        """
        
        for i in range(sync.dhold.considered_tokens_num_merker[0]):
            # case considered token is stop token:
            if torch.any(torch.tensor(sync.dhold.logits_merker[0, 0, i, 0] == sync.mhold.helper.stop_token)):
                sync.dhold.total_probs.append(math.log(sync.dhold.logits_merker[0, 0, i, 1]))
                sync.dhold.prediction_paths_probs.append([math.log(sync.dhold.logits[0, 0, i, 1])])
                sync.dhold.prediction_paths_indices.append([sync.dhold.logits[0, 0, i, 0].to(torch.int32)])
                sync.dhold.skip_path.append(i)
                continue
                
            highest_path_probs = []
            highest_path_indices = []

            for token_num in range(sync.dhold.logits.shape[1]):
                highest_path_probs.append(math.log(sync.dhold.logits[i, token_num, 0, 1]))
                highest_path_indices.append(sync.dhold.logits[i, token_num, 0, 0].to(torch.int32))
                pass
            total_prob = math.log(sync.dhold.logits_merker[0, 0, i, 1])
            total_prob += sum(highest_path_probs)
            sync.dhold.total_probs.append(total_prob)
            sync.dhold.prediction_paths_probs.append([math.log(sync.dhold.logits_merker[0, 0, i, 1])]+highest_path_probs)
            sync.dhold.prediction_paths_indices.append([sync.dhold.logits_merker[0, 0, i, 0].to(torch.int32)]+highest_path_indices)
    def beamsearch_get_best_beam_from_beams(self, sync):
        """
        gets index of the best beam from 
        sync.dhold.total_probs and sets 
        best_beam_probs/indices to the values of that beam
        """
        if sync.dhold.inputs['debugmode']:
            print("paths total probs:", [round(entry, 3) for entry in sync.dhold.total_probs])

        best_beam = max(enumerate(sync.dhold.total_probs),key=lambda x: x[1])[0]
        sync.dhold.best_beam_index = best_beam
        if sync.dhold.inputs['debugmode']: print("chosen best beam:", sync.dhold.best_beam_index)

        sync.dhold.best_beam_probs = sync.dhold.prediction_paths_probs[best_beam]
        sync.dhold.best_beam_indices = torch.tensor(sync.dhold.prediction_paths_indices[best_beam])
        if sync.dhold.inputs['debugmode']: print("sync.dhold.best_beam_indices:", sync.dhold.best_beam_indices)
    def beamsearch_get_returned_content(self, sync):
        """
        decodes the content of the first entry in 
        sync.dhold.inputs['tokens'] and saves it to 
        sync.dhold.returned_content
        """
        
        sync.dhold.returned_content = sync.mhold.helper.decode(sync.dhold.inputs['tokens'][0][sync.dhold.original_input_len:], skip_special_tokens=True)

        # make sure output is 2D for consistency
        if isinstance(sync.dhold.returned_content[0], str):
            sync.dhold.returned_content = [sync.dhold.returned_content]
            
    def beamsearch_check_break_condition(self, sync):
        """
        checks if the best beam has been stopped due to 
        stop token (llama-cpp only),
        otherwise checks if a stop token is in 
        sync.dhold.tokens_to_add. if yes, sets 
        sync.dhold.beamsearch_break = True
        """
        
        sync.dhold.beamsearch_break = False

        if "best_beam_index" in sync.dhold.__dict__ and sync.dhold.stopped[sync.dhold.best_beam_index]:
            sync.dhold.beamsearch_break = True
            print("stop signal in best beam, stopping.")
            

        if sync.dhold.inputs['debugmode']: print("sync.dhold.tokens_to_add:", sync.dhold.tokens_to_add, "torch.tensor(sync.mhold.helper.stop_token):", torch.tensor(sync.mhold.helper.stop_token))
        if torch.any(torch.isin(torch.tensor(sync.dhold.tokens_to_add, device=sync.config['torch_device']), torch.tensor(sync.mhold.helper.stop_token))):
            if sync.dhold.inputs['debugmode']:
                print("tokens to add contained stop token, stopping.")
            sync.dhold.beamsearch_break = True
        
        if sync.dhold.generated_tokens >= sync.dhold.inputs['max_new_tokens']:
            sync.dhold.beamsearch_break = True
            if sync.dhold.inputs['debugmode']:
                print("reached max_new_tokens, stopping.")
    def beamsearch_do_search(self, sync):
        """
        copies sync.dhold.considered_tokens_num to a merker
        if number of considered tokens is 1, will not do 
        beam search.
        otherwise runs sync.get_best_path() which does the 
        beam search and appends the sure_tokens of the best 
        path to the input tokens for the next generation
        """
        
        sync.dhold.considered_tokens_num_merker = copy.deepcopy(sync.dhold.considered_tokens_num)
        if sync.dhold.considered_tokens_num_merker[0] == 1:
            sync.dhold.tokens_to_add = torch.tensor([sync.dhold.logits[0, 0, 0, 0].to(torch.int32)])
            sync.dhold.best_path_indices = sync.dhold.tokens_to_add
            sync.dhold.additional_sure_tokens = 0
            sync.dhold.logits_merker = copy.deepcopy(sync.dhold.logits)
            sync.dhold.best_beam_indices = torch.tensor([sync.dhold.logits_merker[0, 0, 0, 0].to(torch.int32)])
            
        else:
            sync.get_best_path()

            sync.dhold.tokens_to_add = [sync.dhold.inputs['tokens'][sync.dhold.best_beam_index, -1].to(torch.int32)] # at least at the init token for the best path
            if sync.dhold.inputs['debugmode']: print("tokens to add base:", sync.mhold.helper.decode(torch.stack(sync.dhold.tokens_to_add)))
            sync.dhold.additional_sure_tokens = 0
            for i in range(1, len(sync.dhold.best_beam_indices)): # skip 0 since already added
                if sync.dhold.best_beam_probs[i] >= math.log(sync.dhold.inputs['min_conf_for_sure']):
                    sync.dhold.additional_sure_tokens += 1
                    sync.dhold.tokens_to_add.append(sync.dhold.best_beam_indices[i].to(sync.config['torch_device']))
                else:
                    break
            if sync.dhold.inputs['debugmode']: print("tokens to add extended:", sync.mhold.helper.decode(torch.stack(sync.dhold.tokens_to_add)))

            # for llama-cpp: if stop signal in best beam: append whole beam
            # TODO: make it work for when only partial beam appended, stop signal invalid
            if sync.dhold.stopped[sync.dhold.best_beam_index]:
                sync.dhold.tokens_to_add = sync.dhold.best_beam_indices
                
            sync.dhold.tokens_to_add = torch.tensor(sync.dhold.tokens_to_add)

    def build_task(self, sync, instruction, information_input=None):
        """
        builds a prompt for agent task mode
        """
        
        sync.dhold.inputs['chat'] = [chat for chat in sync.dhold.inputs['chat'] if chat['role'] != "System"]
        for i in range(len(sync.dhold.inputs['chat'])):
            old_role = sync.dhold.inputs['chat'][i]['role']
            if old_role == "User": sync.dhold.inputs['chat'][i]['role'] = "user"
            if old_role == "AI": sync.dhold.inputs['chat'][i]['role'] = "assistant"

        instruction_part = instruction
        if information_input == None:
            information_input_part = sync.dhold.inputs['chat'][-1]['content']
        else:
            information_input_part = information_input

        if sync.dhold.inputs['debugmode']: print("instruction_part:", instruction_part, "\ninformation_input_part:", information_input_part)
        sync.dhold.inputs['chat'] = [{'role': 'system', 'content': instruction_part}, {'role': 'user', 'content': information_input_part}]

        if sync.dhold.inputs['debugmode']: print('\nbuilding new prompt string')
        self.build_prompt_string(sync)
