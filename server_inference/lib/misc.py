import os
import json
system_prompt = "You are a helpful assistant."
functions = """You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_web_info", "description": "get_web_info(symbol: str) -> dict - Get web results for a given query.\\n\\n    Args:\\n        query (str): The web search query.\\n\\n    Returns:\\n        dict: A dictionary containing web search results.\\n            Keys:\\n                - \'website\': The first returned website.\\n                - \'website_content\': The content of the website.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}} {"type": "function", "function": {"name": "run_python", "description": "run_python(code: str) -> dict - Returns stdout and stderr from running the provided coda.\\n\\n    Args:\\n        code (str): The python code to run.\\n\\n    Returns:\\n        dict: A dictionary the outputs.\\n            Keys:\\n                - \'stdout\': stdout from running the python code.\\n                - \'stderr\': stderr from running the python code.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>"""

def setup():
    a = """
    with open(f"config.json", "r") as config_file:
        config_tmp = json.loads(config_file.read().strip())
    
    config_tmp['models']['Hermes-2-Theta-Llama-3-8B']['functions'] = functions
    
    with open(f"config.json", "w") as config_file:
        config_file.write(json.dumps(config_tmp, indent=4))
    """

    with open(f"config.json", "r") as config_file:
        config = json.loads(config_file.read().strip())
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"garbage_collection_threshold:{config['torch_cuda_garbage_collection_threshold']}"

    return config

config = setup()

device = config['torch_device']
device_map = config['torch_device_map']

from .imports import *

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

def get_generation_stats(dhold):
    # in case visual tokens have a separate tensor
    if isinstance(dhold.input_shape, list) and not isinstance(dhold.input_shape[0], int):
        dhold.num_input_tokens = 0
        for sublist in dhold.input_shape:
            to_add = 1
            for dim_size in sublist:
                to_add *= dim_size
            dhold.num_input_tokens += to_add
    else:
        dhold.num_input_tokens = 0
        to_add = 1
        for dim_size in dhold.input_shape:
            to_add *= dim_size
        dhold.num_input_tokens += to_add
    dhold.num_output_tokens = 1
    for dim_size in dhold.output_shape:
        dhold.num_output_tokens *= dim_size
    dhold.total_time_taken = round(time.time() - dhold.start_time_total, 2)
    dhold.tokens_per_second = np.round(dhold.num_output_tokens/(time.time() - dhold.start_time_inference), 2)

    dhold.available_mem, dhold.total_mem = torch.cuda.mem_get_info()

def to_GiB(val):
    return round(val/(1024**3), 2)

def make_output_dict_str(sync, show_info=False):
    dhold = sync.dhold
    
    if dhold.error:
        dhold.output_dict = json.dumps({'status': 'error', 'error-info': dhold.error_info})
        return None
        
    if show_info:
        print("dhold.returned_content:", dhold.returned_content, flush=True)

    get_generation_stats(dhold)
    
    dhold.output_dict = json.dumps({'status': 'success', 'returned_content': dhold.returned_content, 'info': {'mem_used':to_GiB(dhold.total_mem-dhold.available_mem), 'mem_total':to_GiB(dhold.total_mem), 'num_input_tokens': dhold.num_input_tokens, 'num_output_tokens': dhold.num_output_tokens, 'total_time_taken': dhold.total_time_taken, 'tokens_per_second': dhold.tokens_per_second}}, default=str)

def prep_for_new_gen(sync, request_json, show_info=False):
    sync.make_new_dhold()
    dhold = sync.dhold
    dhold.request_json = request_json
    
    dhold.start_time_total =  time.time()
    data = dhold.request_json
    print("inputs:")
    counter = 1
    if show_info:
        for key, val in data.items():
            if key != "images":
                if key == "chat":
                    print(" |", key, "(dict):")
                    for entry in val:
                        if counter%2==0:
                            print("    |", f"{str(entry['role']+': ').ljust(14, '.')} {entry['content']}")
                        else:
                            print("    |", f"{str(entry['role']+': ').ljust(14)} {entry['content']}")
                        counter += 1
                else:
                    if not isinstance(val, dict):
                        if counter%2==0:
                            print(" |", str(key+': ').ljust(24, '.'), val, flush=True)
                        else:
                            print(" |", str(key+': ').ljust(24), val, flush=True)
                        counter += 1
                    else:
                        print(" |", key, "(dict):")
                        for sub_key, sub_val in val.items():
                            if counter%2==0:
                                print("    |", str(sub_key+': ').ljust(24, '.'), sub_val, flush=True)
                            else:
                                print("    |", str(sub_key+': ').ljust(24), sub_val, flush=True)
                            counter += 1
                        

    if not isinstance(data['chat'], list):
        data['chat'] = [data['chat']]

    inputs = data
    if len(inputs['images']) > 0:
        inputs['images'] = base64_to_pil(inputs['images'])
        if show_info:
            for image in inputs['images']:
                print(" |", image.size)

    if show_info:
        print()
    inputs['max_new_tokens'] = int(inputs['max_new_tokens'].strip())
    
    dhold.inputs = inputs

def base64_to_pil(base64_images):
    pil_images = []
    for base64_image in base64_images:
        decoded_data = base64.b64decode(base64_image.split(',')[1])
        image_data = io.BytesIO(decoded_data)
        pil_image = Image.open(image_data)
        pil_images.append(pil_image)

    return pil_images


# returns np array: (batch_dim, token_index, logit_index, (token, probability))
def find_top_indexes(sync, arr):
    if arr is None:
        return None
    n_top = sync.dhold.inputs['max_num_beams']
    arr = torch.tensor(arr)
    if len(arr.shape) == 2:
        arr = arr.unsqueeze(0)

    nan_mask = torch.isnan(arr)
    arr[nan_mask] = -float('inf')

    softmax_arr = torch.softmax(arr, dim=-1)
    top_values, top_indexes = torch.topk(softmax_arr, n_top, dim=-1)

    result_probs = top_values
    result_indices = top_indexes

    result = torch.stack([result_indices, result_probs], dim=-1)

    # if not sync.mhold.backend == "llama-cpp" and not sync.mhold.backend == "exllamav2":
    #     result = result.permute(1, 0, 2, 3)
    return result

def test_models(model, test_mode, multi_turn, infer):
    if not isinstance(model, list):
        model = [model]

    if test_mode:
        for entry in model:
            if not multi_turn:
                print("\n\n\n\n\n\n\n")
                infer({
                    'chat': [{'role': 'System', 'content': 'Hello, I am the system.'}, {'role': 'User', 'content': 'Hi!'}], 
                    'model': entry, 
                    'manual_system_prompt': '', 
                    'use_functions': False, 
                    'use_voiceinput': False,
                    'use_voiceoutput': False,
                    'allow_imagegen': False,
                    'agent_task_mode': False,
                    'model_dtype': 'bfloat16', 
                    'max_new_tokens': '256', 
                    'debugmode': True, 
                    'images': [], 
                    'beam_config': {
                        'use_beam_search': True, 
                        'max_num_beams': '2', 
                        'depth_beams': '4', 
                        'min_conf_for_sure': '0.95', 
                        'min_conf_for_consider': '0.02', 
                        'prob_sum_for_search': '0.98'
                    }
                })

class ExLlamaV2_helper():
    def __init__(self, sync, model, cache, tokenizer):
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.sequence_ids = None
        self.abort_event = None
        self.sync = sync

    def _gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           input_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def generate(self, tokens: torch.Tensor, position_offsets, num_tokens):
        assert len(tokens.shape) == 2, f"tokens need to be 2D tensor (batch_dim, tokens), got: {tokens.shape}"
        stop_token = self.tokenizer.eos_token_id
        random.seed(0)
        batch_size = tokens.shape[0]
        loras = None

        logits_merker = []
        tokens_decoded_merker = []
        
        for i in range(num_tokens):

            # Truncate prompt if generation would cause cache overflow
            overflow = tokens.shape[-1] + num_tokens - self.model.config.max_seq_len
            if overflow > 0: tokens = tokens[:, overflow:]
            else: overflow = 0
    
            mask = self.tokenizer.padding_mask(tokens) if batch_size > 1 else None
            first_token = tokens.shape[-1]
    
            self._gen_begin_base(tokens,
                                 mask,
                                 loras,
                                 position_offsets = position_offsets,
                                 input_embeddings = None)

        
            logits = self.model.forward(
                self.sequence_ids[:, -1:],
                self.cache,
                input_mask = mask,
                loras = loras,
                position_offsets = position_offsets,
                indexed_embeddings = None
            ).float().cpu()
    
            logits = torch.softmax(logits, dim=-1)
            logits_merker.append(logits)
            top_token = find_top_indexes(self.sync, logits.numpy(), n_top=1)[0, 0, 0, 0]
            tokens = torch.concatenate([tokens, torch.tensor([top_token.astype(np.int32)]).to(self.sync.config['torch_device']).unsqueeze(0)], dim=-1)
            top_token_decoded = self.sync.mhold.helper.decode(torch.tensor([top_token.astype(np.int32)]), decode_special_tokens=True)[0]
            
            if torch.tensor([top_token.astype(np.int32)]) in self.sync.mhold.stop_token: 
                break
            tokens_decoded_merker.append(top_token_decoded)

        return tokens_decoded_merker, find_top_indexes(self.sync, torch.concatenate(logits_merker, dim=1).numpy(), n_top=self.sync.dhold.inputs['max_num_beams'])

    

    def encode(self, inputs: Union[str, List[str]], encode_special_tokens=True):
        if isinstance(inputs, str):
            inputs = [inputs]

        # ids_merker = []
        # position_offsets_merker = []
        # for entry in inputs:
        tokens, position_offsets = self.tokenizer.encode(
            inputs,
            encode_special_tokens = encode_special_tokens,
            return_offsets = True,
            add_bos = False)
        # ids_merker.append(ids)
        # position_offsets_merker.append(position_offsets)

        # tokens = torch.stack(ids_merker)
        # position_offsets = torch.stack(position_offsets_merker)
        return tokens, position_offsets

    def decode(self, inputs: torch.Tensor, decode_special_tokens=False, logits_separate=False):
        if isinstance(inputs, list):
            inputs = torch.tensor(inputs)
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        assert (len(inputs.shape) == 2 or len(inputs.shape) == 1), f"inputs for decode should be 1D or 2D tensor, got: {inputs.shape}"


        decoded_strings = []
        for entry in inputs:
            if logits_separate:
                decoded_strings.append([])
                for sample in entry:
                    decoded = self.tokenizer.decode(torch.tensor([sample]), decode_special_tokens=decode_special_tokens)
                    decoded_strings[-1].append(decoded)
            else:
                decoded = self.tokenizer.decode(entry, decode_special_tokens=decode_special_tokens)
                decoded_strings.append(decoded)

        if logits_separate: 
            decoded_strings = decoded_strings[0]
        return decoded_strings


def show_dict_compact(dict_input, indent=0):
    indentation = "   "*indent
    if isinstance(dict_input, list):
        if len(dict_input) <= 20:
            for entry in dict_input:
                if isinstance(entry, dict) or isinstance(entry, list):
                    show_dict_compact(entry, indent=indent+1)
                elif isinstance(entry, torch.Tensor) or isinstance(entry, np.ndarray):
                    print(indentation, type(entry), entry.shape)
                else:
                    print(indentation, type(entry))
        else:
            print(indentation, type(dict_input), f"len: {len(dict_input)}")

    elif isinstance(dict_input, dict):
        if len(dict_input) <= 20:
            for key, val in dict_input.items():
                if isinstance(val, dict) or isinstance(val, list):
                    print(indentation, f"{key}:")
                    show_dict_compact(val, indent=indent+1)
                elif isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
                    print(indentation, key, type(val), val.shape)
                else:
                    print(indentation, key, type(val))
        else:
            print(indentation, type(dict_input), f"len: {len(dict_input)}")
    else:
        print(f"Error: input is neither dict nor list, got: {type(dict_input)}")
        
            





