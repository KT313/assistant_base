import os
# garbage_collection_threshold between 0 and 1 in % of mem
# backend native or cudaMallocAsync
# PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2>:<value2>...
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.01"

import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from conversation import conv_templates, SeparatorStyle

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from PIL import Image
import requests
import torch
import copy
import json
import time
import gc

device = "cuda"
device_map = "auto"
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"

class Sync():
    def __init__(self):
        self.current_model = ""

    def prep_gen_inputs(self, args):
        # args should contain:
        # model
        # chat
        # image

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
            
            else:
                image = Image.open(args['image'])
                image_tensor = process_images([image], self.image_processor, self.model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
                
                question = DEFAULT_IMAGE_TOKEN + f"\n{args['chat'][-1]['content']}"
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()                
                tokens = tokenizer_image_token(prompt_question, sync.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                image_sizes = [image.size]

                self.gen_inputs = {}
                self.gen_inputs['tokens'] = tokens
                self.gen_inputs['image_tensor'] = image_tensor
                self.gen_inputs['image_sizes'] = image_sizes
                
            
        if args['model'] == "paligemma-3b-mix-448":
            if args['image'] == None:
                return {'error': 'error', 'error-info':'paligemma-3b-mix-448 only works if an image is provided'}
            else:
                prompt = args['chat'][-1]['content']
                tokens = self.processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                self.gen_inputs = {}
                self.gen_inputs['tokens'] = tokens

        return self.gen_inputs

    def generate(self, args):
        # args should contain:
        # model
        # gen_inputs

        # llama3-llava-next-8b
        if args['model'] == "llama3-llava-next-8b":
            # with image
            if args['gen_inputs']['image_tensor'] != None:  
                pass
            else:
                cont = sync.model.generate(
                    args['gen_inputs']['tokens'],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                )

            output_shape = cont.shape
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            returned_content = [entry.strip() for entry in text_outputs]

        if args['model'] == "paligemma-3b-mix-448":
            input_len = args['gen_inputs']['tokens']['input_ids'].shape[-1]
            generation = self.model.generate(**args['gen_inputs']['tokens'], max_new_tokens=100, do_sample=False)
            output_shape = generation.shape
            generation = generation[0][input_len:]
            text_outputs = [self.processor.decode(generation, skip_special_tokens=True)]
            returned_content = [entry.strip() for entry in text_outputs]
        
        return returned_content, output_shape
















        

sync = Sync()

def load_model(model_name, sync):
    if model_name == "llama3-llava-next-8b":
        pretrained = "../../../models/multimodal/llama3-llava-next-8b"
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
        pretrained = "../../../models/multimodal/paligemma-3b-mix-448"
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
        


            gen_inputs = sync.prep_gen_inputs({'model': data['model'], 'chat': data['chat'], 'image': None})
            print("gen_inputs:", gen_inputs, flush=True)
            if "error" in gen_inputs:
                return json.dumps({'status': 'error', 'error-info': gen_inputs['error-info']})
            start_time_inference = time.time()
            gen_outputs, output_shape = sync.generate({'model': data['model'], 'gen_inputs': gen_inputs})
            print("gen_outputs:", gen_outputs, flush=True)

            
        
            # TODO add info about num input tokens, output tokens, tokens/s speed
            num_input_tokens = 1
            for dim_size in gen_inputs['tokens'].shape:
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