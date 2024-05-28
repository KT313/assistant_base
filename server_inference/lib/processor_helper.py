from .imports import *

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