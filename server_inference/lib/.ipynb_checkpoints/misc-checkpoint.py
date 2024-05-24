import os
import json

functions = """You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_web_info", "description": "get_web_info(symbol: str) -> dict - Get web results for a given query.\\n\\n    Args:\\n        query (str): The web search query.\\n\\n    Returns:\\n        dict: A dictionary containing web search results.\\n            Keys:\\n                - \'website\': The first returned website.\\n                - \'website_content\': The content of the website.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}} {"type": "function", "function": {"name": "run_python", "description": "run_python(code: str) -> dict - Returns stdout and stderr from running the provided coda.\\n\\n    Args:\\n        code (str): The python code to run.\\n\\n    Returns:\\n        dict: A dictionary the outputs.\\n            Keys:\\n                - \'stdout\': stdout from running the python code.\\n                - \'stderr\': stderr from running the python code.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>"""

def setup():
    with open(f"config.json", "r") as config_file:
        config_tmp = json.loads(config_file.read().strip())
    
    config_tmp['models']['Hermes-2-Theta-Llama-3-8B']['functions'] = functions
    
    with open(f"config.json", "w") as config_file:
        config_file.write(json.dumps(config_tmp, indent=4))

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
    dhold.num_input_tokens = 1
    for dim_size in dhold.input_shape:
        dhold.num_input_tokens *= dim_size
    dhold.num_output_tokens = 1
    for dim_size in dhold.output_shape:
        dhold.num_output_tokens *= dim_size
    dhold.total_time_taken = round(time.time() - dhold.start_time_total, 2)
    dhold.tokens_per_second = round(dhold.num_output_tokens/(time.time() - dhold.start_time_inference), 2)

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
    
    dhold.output_dict = json.dumps({'status': 'success', 'returned_content': dhold.returned_content, 'info': {'mem_used':to_GiB(dhold.total_mem-dhold.available_mem), 'mem_total':to_GiB(dhold.total_mem), 'num_input_tokens': dhold.num_input_tokens, 'num_output_tokens': dhold.num_output_tokens, 'total_time_taken': dhold.total_time_taken, 'tokens_per_second': dhold.tokens_per_second}})

def prep_for_new_gen(sync, request_json, show_info=False):
    sync.make_new_dhold()
    dhold = sync.dhold
    dhold.request_json = request_json
    
    dhold.start_time_total =  time.time()
    data = dhold.request_json
    if show_info:
        print("data:", data, flush=True)

    if not isinstance(data['chat'], list):
        data['chat'] = [data['chat']]

    inputs = data
    inputs['image'] = None
    inputs['max_new_tokens'] = int(inputs['max_new_tokens'].strip())
    
    dhold.inputs = inputs