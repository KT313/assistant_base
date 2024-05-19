import os
import json
with open(f"config.json", "r") as config_file:
    config = json.loads(config_file.read().strip())

# garbage_collection_threshold between 0 and 1 in % of mem
# backend native or cudaMallocAsync
# PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2>:<value2>...
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"garbage_collection_threshold:{config['torch_cuda_garbage_collection_threshold']}"

import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from conversation import conv_templates, SeparatorStyle

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from llama_cpp import Llama

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import bitsandbytes, flash_attn

from PIL import Image
import requests
import torch
import copy
import time
import gc

device = config['torch_device']
device_map = config['torch_device_map']

class Sync():
    def __init__(self):
        self.current_model = ""

    def prep_gen_inputs(self, args):
        # args should contain:
        # model
        # chat
        # image

        args['chat'] = [chat for chat in args['chat'] if chat['role'] != "System"]
        
        self.error = False
        self.error_info = ""

        # check if model needs to be changed
        print(args['model'], self.current_model, flush=True)
        if args['model'] != self.current_model:
            load_model(args['model'], self)

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
                tokens = tokenizer_image_token(prompt_question, self.tokenizer, return_tensors="pt").unsqueeze(0).to(device)
                
                self.gen_inputs = {}
                self.gen_inputs['tokens'] = tokens
                self.gen_inputs['image_tensor'] = None
                self.gen_inputs['image_sizes'] = None
                self.input_shape = self.gen_inputs['tokens'].shape
            
            else:
                image = Image.open(args['image'])
                image_tensor = process_images([image], self.image_processor, self.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
                
                question = DEFAULT_IMAGE_TOKEN + f"\n{args['chat'][-1]['content']}"
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()                
                tokens = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                image_sizes = [image.size]

                self.gen_inputs = {}
                self.gen_inputs['tokens'] = tokens
                self.gen_inputs['image_tensor'] = image_tensor
                self.gen_inputs['image_sizes'] = image_sizes
                self.input_shape = self.gen_inputs['tokens'].shape
                
            
        if args['model'] == "paligemma-3b-mix-448":
            if args['image'] == None:
                self.error = True
                self.error_info = "paligemma-3b-mix-448 only works if an image is provided"
                return None
            else:
                prompt = args['chat'][-1]['content']
                tokens = self.processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                self.gen_inputs = {}
                self.gen_inputs['tokens'] = tokens
                self.input_shape = self.gen_inputs['tokens'].shape

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

            self.gen_inputs = {}
            self.gen_inputs['text'] = prompt_question

        if args['model'] == "Hermes-2-Theta-Llama-3-8B":
            new_chat = []
            if "system_prompt" in config['models'][self.current_model]:
                new_chat.append({'role': 'system', 'content': config['models'][self.current_model]['system_prompt']})
            for entry in args['chat']:
                if entry['role'] == "User":
                    new_chat.append({'role': 'user', 'content': entry['content']})
                if entry['role'] == "AI":
                    new_chat.append({'role': 'assistant', 'content': entry['content']})
            args['chat'] = new_chat
            
            tokens = self.tokenizer.apply_chat_template(args['chat'], return_tensors="pt").to(device)
            
            self.gen_inputs = {}
            self.gen_inputs['tokens'] = tokens
            self.input_shape = self.gen_inputs['tokens'].shape





        self.gen_inputs['model'] = args['model']

    def generate(self):
        args = self.gen_inputs
        # args should contain:
        # model
        # gen_inputs

        # llama3-llava-next-8b
        # print(json.dumps(args, indent=4), flush=True)
        if args['model'] == "llama3-llava-next-8b":
            if args['image_tensor'] != None:  
                pass
            else:
                cont = self.model.generate(
                    args['tokens'],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=config['max_new_tokens'],
                )

            self.output_shape = cont.shape
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            self.returned_content = [entry.strip() for entry in text_outputs]
            print("\n\nself.returned_content:", self.returned_content, "\n\n")

        if args['model'] == "paligemma-3b-mix-448":
            input_len = args['tokens']['input_ids'].shape[-1]
            generation = self.model.generate(**args['tokens'], max_new_tokens=config['max_new_tokens'], do_sample=False)
            self.output_shape = generation.shape
            generation = generation[0][input_len:]
            text_outputs = [self.processor.decode(generation, skip_special_tokens=True)]
            self.returned_content = [entry.strip() for entry in text_outputs]

        if args['model'] == "Meta-Llama-3-70B-Instruct-IQ2_S" or args['model'] == "Meta-Llama-3-70B-Instruct-IQ1_M":
            output = self.model(
              args['text'], # Prompt
              max_tokens=config['max_new_tokens'], # Generate up to 32 tokens, set to None to generate up to the end of the context window
              stop=["<|eot_id|>", "<|end_of_text|>"], # Stop generating just before the model would generate a new question
              echo=False # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion
            self.returned_content = [out['text'] for out in output['choices']]
            self.output_shape = [1, output['usage']['completion_tokens']]
            self.input_shape = [1, output['usage']['prompt_tokens']]

        if args['model'] == "Hermes-2-Theta-Llama-3-8B":
            print(args['tokens'], flush=True)
            print(self.tokenizer.decode(args['tokens'][0], skip_special_tokens=False, clean_up_tokenization_space=True), flush=True)
            generated_ids = self.model.generate(args['tokens'], max_new_tokens=config['max_new_tokens'], temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(generated_ids[0][args['tokens'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
            self.output_shape = generated_ids[0][args['tokens'].shape[-1]:].shape
            self.returned_content = [response.strip()]












        

sync = Sync()

def load_model(model_name, sync):
    try:
        del sync.model
        gc.collect()
    except:
        pass
        
    if model_name == "llama3-llava-next-8b":
        pretrained = config['models'][model_name]['path']
        model_name_for_loading = "llava_llama3"
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name_for_loading, device_map=device_map) # Add any other thing you want to pass in llava_model_args
        
        model.eval()
        model.tie_weights()
        
        sync.tokenizer = tokenizer
        sync.model = model
        sync.image_processor = image_processor
        sync.max_length = max_length
        sync.current_model = model_name

    if model_name == "paligemma-3b-mix-448":
        pretrained = config['models'][model_name]['path']
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            revision="bfloat16",
        ).eval()
        processor = AutoProcessor.from_pretrained(pretrained)

        sync.processor = processor
        sync.model = model
        sync.current_model = model_name

    if model_name == "Meta-Llama-3-70B-Instruct-IQ2_S":
        pretrained = config['models'][model_name]['path']
        model = Llama(
            model_path=pretrained,
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            n_ctx=1024, # Uncomment to increase the context window
            flash_attn=True,
        )
        sync.model = model
        sync.current_model = model_name

    if model_name == "Meta-Llama-3-70B-Instruct-IQ1_M":
        pretrained = config['models'][model_name]['path']
        model = Llama(
            model_path=pretrained,
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            n_ctx=1024, # Uncomment to increase the context window
            flash_attn=True,
        )
        sync.model = model
        sync.current_model = model_name

    if model_name == "Hermes-2-Theta-Llama-3-8B":
        pretrained = config['models'][model_name]['path']
        tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=False)
        model = LlamaForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.float16,
            device_map=config['torch_device_map'],
            load_in_8bit=False,
            load_in_4bit=False,
            use_flash_attention_2=True
        )
        
        sync.tokenizer = tokenizer
        sync.model = model
        sync.current_model = model_name







        

# load_model("llama3-llava-next-8b", sync)



import weave

### wandb stuff ###

from weave.weave_init import *
from weave import *

project_name = "weave-web-test"
ensure_project_exists = True

wandb_api.init()
wandb_context = wandb_api.get_wandb_api_context()
if wandb_context is None:
    import wandb
    print("Please login to Weights & Biases (https://wandb.ai/) to continue:")
    wandb.login(anonymous="never", force=True)
    wandb_api.init()
    wandb_context = wandb_api.get_wandb_api_context()

entity_name, project_name = get_entity_project_from_project_name(project_name)
wandb_run_id = weave_client.safe_current_wb_run_id()

api_key = "e178771e66d0b5e45981c669d427c3ea2703d33b"
remote_server = init_weave_get_server(api_key)
client = weave_client.WeaveClient(entity_name, project_name, remote_server, ensure_project_exists)

init_client = InitializedClient(client)

autopatch.autopatch()

username = get_username()
try:
    min_required_version = (remote_server.server_info().min_required_weave_python_version)
except Exception:
    min_required_version = "0.0.0"
init_message.assert_min_weave_version(min_required_version)
init_message.print_init_message(username, entity_name, project_name, read_only=not ensure_project_exists)

user_context = {"username": username} if username else None
trace_sentry.global_trace_sentry.configure_scope({"entity_name": entity_name,"project_name": project_name,"user":user_context,})

###################



class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

@weave.op()
def infer_helper(request_json):
    with ClearCache():
        with torch.no_grad():
            start_time_total =  time.time()
            data = request_json
            print("\n\n\n")
            print("data:", request_json, flush=True)
            print("\n\n\n")

            if not isinstance(data['chat'], list):
                data['chat'] = [data['chat']]
        


            sync.prep_gen_inputs({'model': data['model'], 'chat': data['chat'], 'image': None})
            
            if sync.error:
                return json.dumps({'status': 'error', 'error-info': sync.error_info})
            start_time_inference = time.time()
            sync.generate()
            
            gen_outputs, output_shape, input_shape = sync.returned_content, sync.output_shape, sync.input_shape
            
            print("gen_outputs:", gen_outputs, flush=True)

            
        
            # TODO add info about num input tokens, output tokens, tokens/s speed
            num_input_tokens = 1
            for dim_size in input_shape:
                num_input_tokens *= dim_size
            num_output_tokens = 1
            for dim_size in output_shape:
                num_output_tokens *= dim_size
            total_time_taken = round(time.time() - start_time_total, 2)
            tokens_per_second = round(num_output_tokens/(time.time() - start_time_inference), 2)
        
            available_mem, total_mem = torch.cuda.mem_get_info()
        
            def to_GiB(val):
                return round(val/(1024**3), 2)
            
            return json.dumps({'status': 'success', 'returned_content': gen_outputs, 'info': {'mem_used':to_GiB(total_mem-available_mem), 'mem_total':to_GiB(total_mem), 'num_input_tokens': num_input_tokens, 'num_output_tokens': num_output_tokens, 'total_time_taken': total_time_taken, 'tokens_per_second': tokens_per_second}})



from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    init_client = InitializedClient(client)
    return infer_helper(request.get_json())

if __name__ == '__main__':
    app.run(port=10000)