import os
import json

functions = """You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_web_info", "description": "get_web_info(symbol: str) -> dict - Get web results for a given query.\\n\\n    Args:\\n        query (str): The web search query.\\n\\n    Returns:\\n        dict: A dictionary containing web search results.\\n            Keys:\\n                - \'website\': The first returned website.\\n                - \'website_content\': The content of the website.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}} {"type": "function", "function": {"name": "run_python", "description": "run_python(code: str) -> dict - Returns stdout and stderr from running the provided coda.\\n\\n    Args:\\n        code (str): The python code to run.\\n\\n    Returns:\\n        dict: A dictionary the outputs.\\n            Keys:\\n                - \'stdout\': stdout from running the python code.\\n                - \'stderr\': stderr from running the python code.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>"""

with open(f"config.json", "r") as config_file:
    config_tmp = json.loads(config_file.read().strip())

config_tmp['models']['Hermes-2-Theta-Llama-3-8B']['functions'] = functions

with open(f"config.json", "w") as config_file:
    config_file.write(json.dumps(config_tmp, indent=4))








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

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig
import bitsandbytes, flash_attn

from PIL import Image
import requests
import torch
import copy
import time
import math
import gc

device = config['torch_device']
device_map = config['torch_device_map']

class Sync():
    def __init__(self):
        self.current_model = ""
        self.current_dtype = ""

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
        if args['model'] != self.current_model or args['model_dtype'] != self.current_dtype:
            load_model(args['model'], args['model_dtype'], self)

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
            if args['use_functions'] and "functions" in config['models'][self.current_model]:
                new_chat.append({'role': 'system', 'content': config['models'][self.current_model]['functions']})
            elif 'manual_system_prompt' in args and args['manual_system_prompt'].strip() != "":
                new_chat.append({'role': 'system', 'content': args['manual_system_prompt']})
            elif "system_prompt" in config['models'][self.current_model]:
                new_chat.append({'role': 'system', 'content': config['models'][self.current_model]['system_prompt']})
            
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

            print(chat_string, flush=True)
            
            tokens = self.tokenizer(chat_string, return_tensors="pt").input_ids.to(device)
            
            self.gen_inputs = {}
            self.gen_inputs['tokens'] = tokens
            self.gen_inputs['beam_config'] = args['beam_config']
            self.input_shape = self.gen_inputs['tokens'].shape

        self.gen_inputs['model'] = args['model']



    def get_best_path(self, args, considered_tokens_probs, considered_tokens_indices):
        total_probs  = []
        prediction_paths_probs = []
        prediction_paths_indices = []

        skip_path = []

        batched_input_tokens = []
        
        for i in range(len(considered_tokens_probs)):
            batched_input_tokens.append(self.tokenizer.decode(torch.concatenate((args['tokens'], torch.tensor([[considered_tokens_indices[i]]]).to(device)), dim=-1)[0], skip_special_tokens=False, clean_up_tokenization_space=True))

        batched_input_tokens = self.tokenizer(batched_input_tokens, return_tensors="pt")
                
            
        beam_output = self.model.generate(
            batched_input_tokens.input_ids.to(device),
            attention_mask = batched_input_tokens.attention_mask.to(device),
            max_new_tokens=args['depth_beams'],
            temperature=1.0,
            repetition_penalty=1.1,
            do_sample=False,
            num_beams=1,
            # early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id = 128003
        )

        for i in range(len(considered_tokens_probs)):
            # case considered token is stop token:
            if considered_tokens_indices[i] == 128003:
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

        print("paths total probs:", [round(entry, 3) for entry in total_probs])

        best_beam = max(enumerate(total_probs),key=lambda x: x[1])[0]

        return prediction_paths_probs[best_beam], prediction_paths_indices[best_beam]

    

    def generate(self):
        args = self.gen_inputs
        # args should contain:
        # model
        # gen_inputs

        # llama3-llava-next-8b
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
            )
            self.returned_content = [out['text'] for out in output['choices']]
            self.output_shape = [1, output['usage']['completion_tokens']]
            self.input_shape = [1, output['usage']['prompt_tokens']]

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
            attn_mask = torch.ones_like(args['tokens']).to(device)

            num_beams_this_run = max_num_beams

            print("input:", self.tokenizer.decode(args['tokens'][0], skip_special_tokens=False, clean_up_tokenization_space=True))

            if not args['beam_config']['use_beam_search']:
                output = self.model.generate(
                    args['tokens'],
                    attention_mask = attn_mask,
                    max_new_tokens=config['max_new_tokens'],
                    temperature=0.7,
                    repetition_penalty=1.1,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id = 128003
                )
                response = self.tokenizer.decode(output.sequences[0][original_input_len:], skip_special_tokens=True, clean_up_tokenization_space=True)
                self.output_shape = output.sequences[0][original_input_len:].shape
                self.returned_content = [response.strip()]
                
                return None

            
            
            for step in range(config['max_new_tokens']):

                # custom beam search                
                output = self.model.generate(
                    args['tokens'],
                    attention_mask = attn_mask,
                    max_new_tokens=1,
                    temperature=1.0,
                    repetition_penalty=1.1,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id = 128003
                )
    
                probabilities, indices = torch.topk(torch.softmax(output.scores[0], dim=-1), k=8)
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
                    best_path_probs, best_path_indices = self.get_best_path(args, considered_tokens_probs, considered_tokens_indices)
        
                    tokens_to_add = [best_path_indices[0]] # at least at the init token for the best path
                    additional_sure_tokens = 0
                    for i in range(1, len(best_path_indices)): # skip 0 since already added
                        if best_path_probs[i] >= math.log(min_conf_for_sure):
                            additional_sure_tokens += 1
                            tokens_to_add.append(best_path_indices[i])
                        else:
                            break
                        
                generated_tokens += len(tokens_to_add)
                
                args['tokens'] = torch.concatenate((args['tokens'], torch.tensor(tokens_to_add).unsqueeze(0).to(device)), dim=-1)
                attn_mask = torch.ones_like(args['tokens']).to(device)

                print(" | ".join([str(round(entry, 5)).ljust(14) for entry in probabilities[0].tolist()]))
                print(" | ".join([self.tokenizer.decode(entry, skip_special_tokens=False, clean_up_tokenization_space=True).strip().ljust(14) for entry in indices[0].tolist()]))
                if len(considered_tokens_indices) == 1:
                    print("-> single considered token, not doing beam search")
                else:
                    print(f"-> using {len(considered_tokens_probs)} beams")

                print("\n")
                print(f"current generation: {self.tokenizer.decode(args['tokens'][0][original_input_len:-len(tokens_to_add)], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[32m{self.tokenizer.decode(tokens_to_add, skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m \x1b[37m{self.tokenizer.decode(best_path_indices[1+additional_sure_tokens:], skip_special_tokens=False, clean_up_tokenization_space=True)}\x1b[0m") # \[90m or \[37m for gray \x1b[43
                print("\n\n\n")

                if 128003 in tokens_to_add:
                    print("tokens to add contained stop token, stopping.")
                    break

                if generated_tokens >= config['max_new_tokens']:
                    print("reached max_new_tokens, stopping.")
                    break
            
            print("\n\n\n")

            response = self.tokenizer.decode(args['tokens'][0][original_input_len:], skip_special_tokens=True, clean_up_tokenization_space=True)
            self.output_shape = args['tokens'][0][original_input_len:].shape
            self.returned_content = [response.strip()]













        

sync = Sync()

def load_model(model_name, dtype, sync):
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

    if model_name == "Hermes-2-Theta-Llama-3-8B":
        pretrained = config['models'][model_name]['path']
        tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=False, padding_side='left')

        if dtype == "float16":
            model = LlamaForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.float16,
                device_map=config['torch_device_map'],
                # quantization_config=bnb_config,
                use_flash_attention_2=True
            )
        elif dtype == "bfloat16":
            model = LlamaForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=config['torch_device_map'],
                # quantization_config=bnb_config,
                use_flash_attention_2=True
            )
        else:
            if dtype == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            elif dtype == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            model = LlamaForCausalLM.from_pretrained(
                pretrained,
                # torch_dtype=torch.float16,
                device_map=config['torch_device_map'],
                quantization_config=bnb_config,
                use_flash_attention_2=True
            )
        
        sync.tokenizer = tokenizer
        sync.model = model
        
    sync.current_model = model_name
    sync.current_dtype = dtype







        

# load_model("Hermes-2-Theta-Llama-3-8B", sync)



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
        
            inputs = data
            inputs['image'] = None

            sync.prep_gen_inputs(inputs)
            
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