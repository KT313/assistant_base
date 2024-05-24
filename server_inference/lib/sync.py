from .imports import *
from .model_holder import ModelHolder
from .data_holder import DataHolder

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

        args['chat'] = [chat for chat in args['chat'] if chat['role'] != "System"]

        self.dhold.gen_inputs = args

        # check if model needs to be changed
        if args['debugmode']:
            print(args['model'], self.mhold.current_model, flush=True)
        if self.mhold == None or args['model'] != self.mhold.current_model or args['model_dtype'] != self.mhold.current_dtype:
            self.mhold = ModelHolder()
            self.mhold.load_model(self, args['model'], args['model_dtype'])

        if args['model'] == "llama3-llava-next-8b":
            conv_template = "llava_llama_3"
            conv = copy.deepcopy(conv_templates[conv_template])
            for entry in args['chat'][:-1]:
                if entry['role'] == "User":
                    conv.append_message(conv.roles[0], entry['content'])
                elif entry['role'] == "AI":
                    conv.append_message(conv.roles[1], entry['content'])

            if args['image'] == None:
                question = args['chat'][-1]['content']
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()                
                tokens = tokenizer_image_token(prompt_question, self.mhold.tokenizer, return_tensors="pt").unsqueeze(0).to(self.config['torch_device'])
                
                self.dhold.gen_inputs['tokens'] = tokens
                self.dhold.gen_inputs['image_tensor'] = None
                self.dhold.gen_inputs['image_sizes'] = None
                self.dhold.input_shape = self.dhold.gen_inputs['tokens'].shape
            
            else:
                image = Image.open(args['image'])
                image_tensor = process_images([image], self.mhold.image_processor, self.mhold.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=self.config['torch_device']) for _image in image_tensor]
                
                question = DEFAULT_IMAGE_TOKEN + f"\n{args['chat'][-1]['content']}"
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()                
                tokens = tokenizer_image_token(prompt_question, self.mhold.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.config['torch_device'])
                image_sizes = [image.size]

                self.dhold.gen_inputs['tokens'] = tokens
                self.dhold.gen_inputs['image_tensor'] = image_tensor
                self.dhold.gen_inputs['image_sizes'] = image_sizes
                self.dhold.input_shape = self.dhold.gen_inputs['tokens'].shape
                
            
        if args['model'] == "paligemma-3b-mix-448":
            if args['image'] == None:
                self.dhold.error = True
                self.dhold.error_info = "paligemma-3b-mix-448 only works if an image is provided"
                return None
            else:
                prompt = args['chat'][-1]['content']
                tokens = self.mhold.processor(text=prompt, images=image, return_tensors="pt").to(self.config['torch_device'])

                self.dhold.gen_inputs['tokens'] = tokens
                self.dhold.input_shape = self.dhold.gen_inputs['tokens'].shape

        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
            conv_template = "llama_3_70b"
            conv = copy.deepcopy(conv_templates[conv_template])
            for entry in args['chat'][:-1]:
                if entry['role'] == "User":
                    conv.append_message(conv.roles[0], entry['content'])
                elif entry['role'] == "AI":
                    conv.append_message(conv.roles[1], entry['content'])
            question = args['chat'][-1]['content']
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            self.dhold.gen_inputs['text'] = prompt_question

        if args['model'] == "Hermes-2-Theta-Llama-3-8B":
            new_chat = []
            if args['use_functions'] and "functions" in self.config['models'][self.mhold.current_model]:
                new_chat.append({'role': 'system', 'content': self.config['models'][self.mhold.current_model]['functions']})
            elif 'manual_system_prompt' in args and args['manual_system_prompt'].strip() != "":
                new_chat.append({'role': 'system', 'content': args['manual_system_prompt']})
            elif "system_prompt" in self.config['models'][self.mhold.current_model]:
                new_chat.append({'role': 'system', 'content': self.config['models'][self.mhold.current_model]['system_prompt']})
            
            for entry in args['chat']:
                if entry['role'] == "User":
                    new_chat.append({'role': 'user', 'content': entry['content']})
                if entry['role'] == "AI":
                    new_chat.append({'role': 'assistant', 'content': entry['content']})

            args['chat'] = new_chat

            chat_string = ""
            
            for entry in args['chat']:
                chat_string += f"<|im_start|>{entry['role']}\n"
                chat_string += f"{entry['content']}<|im_end|>\n"
            chat_string += f"<|im_start|>assistant\n"

            if args['debugmode']:
                print(chat_string, flush=True)
            
            tokens = self.mhold.tokenizer(chat_string, return_tensors="pt").input_ids.to(self.config['torch_device'])
            
            self.dhold.gen_inputs['tokens'] = tokens
            self.dhold.gen_inputs['beam_config'] = args['beam_config']
            self.dhold.input_shape = self.dhold.gen_inputs['tokens'].shape

        if args['model'] == "phi-3-vision-128k-instruct":
            if len(args['images']) > 0:
                to_add_for_images = ""
                for i in range(len(args['images'])):
                    to_add_for_images += f"<|image_{i+1}|>\n"
                args['chat'][-1]['content'] = to_add_for_images+args['chat'][-1]['content']
            prompt = self.mhold.processor.tokenizer.apply_chat_template(args['chat'], tokenize=False, add_generation_prompt=True)

            print("prompt:", prompt)

            if len(args['images']) > 0:
                tokens = self.mhold.processor(prompt, args['images'], return_tensors="pt").to(self.config['torch_device'])
            else:
                tokens = self.mhold.processor(prompt, return_tensors="pt").to(self.config['torch_device'])

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

    def generate(self):
        self.dhold.start_time_inference = time.time()
        if self.dhold.error:
            return None
        args = self.dhold.gen_inputs
        # args should contain:
        # model
        # gen_inputs

        # llama3-llava-next-8b
        if args['model'] == "llama3-llava-next-8b":
            if args['image_tensor'] != None:  
                pass
            else:
                cont = self.mhold.model.generate(
                    args['tokens'],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=args['max_new_tokens'],
                )

            self.dhold.output_shape = cont.shape
            text_outputs = self.mhold.tokenizer.batch_decode(cont, skip_special_tokens=True)
            self.dhold.returned_content = [entry.strip() for entry in text_outputs]
            if args['debugmode']:
                print("\n\nself.dhold.returned_content:", self.dhold.returned_content, "\n\n")

        if args['model'] == "paligemma-3b-mix-448":
            input_len = args['tokens']['input_ids'].shape[-1]
            generation = self.mhold.model.generate(**args['tokens'], max_new_tokens=args['max_new_tokens'], do_sample=False)
            self.dhold.output_shape = generation.shape
            generation = generation[0][input_len:]
            text_outputs = [self.mhold.processor.decode(generation, skip_special_tokens=True)]
            self.dhold.returned_content = [entry.strip() for entry in text_outputs]

        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
            output = self.mhold.model(
              args['text'], # Prompt
              max_tokens=args['max_new_tokens'],
              stop=["<|eot_id|>", "<|end_of_text|>"],
              echo=False
            )
            self.dhold.returned_content = [out['text'] for out in output['choices']]
            self.dhold.output_shape = [1, output['usage']['completion_tokens']]
            self.dhold.input_shape = [1, output['usage']['prompt_tokens']]

        if args['model'] == "Hermes-2-Theta-Llama-3-8B":

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

            original_input_len = args['tokens'].shape[-1]
            attn_mask = torch.ones_like(args['tokens'], device=self.config['torch_device'])

            reached_token_limit = False

            num_beams_this_run = max_num_beams
            
            if args['debugmode']:
                print("input:", self.mhold.tokenizer.decode(args['tokens'][0], skip_special_tokens=False, clean_up_tokenization_space=True))

            if not args['beam_config']['use_beam_search']:
                output = self.mhold.model.generate(
                    args['tokens'],
                    attention_mask = attn_mask,
                    max_new_tokens=args['max_new_tokens'],
                    temperature=1.0,
                    repetition_penalty=1.1,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.mhold.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id = 128003
                )
                response = self.mhold.tokenizer.decode(output.sequences[0][original_input_len:], skip_special_tokens=True, clean_up_tokenization_space=True)
                self.dhold.output_shape = output.sequences[0][original_input_len:].shape
                self.dhold.returned_content = [response.strip()]
                
                return None

            
            
            for step in range(args['max_new_tokens']):

                # custom beam search                
                output = self.mhold.model.generate(
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
    
                probabilities, indices = torch.topk(torch.softmax(output.scores[0].detach(), dim=-1), k=8)
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

        if args['model'] == "phi-3-vision-128k-instruct":
            generation_args = { 
                "max_new_tokens": args['max_new_tokens'], 
                "temperature": 1.0, 
                "do_sample": False, 
            } 
            
            generate_ids = self.mhold.model.generate(**args['tokens'], eos_token_id=self.mhold.processor.tokenizer.eos_token_id, **generation_args) 
            
            # remove input tokens 
            generate_ids = generate_ids[:, args['tokens']['input_ids'].shape[1]:]
            response = self.mhold.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            
            self.dhold.output_shape = generate_ids.shape
            self.dhold.returned_content = [response.strip()]

    def make_new_dhold(self):
        self.dhold = DataHolder()