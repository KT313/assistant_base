from lib.misc import *
from lib.imports import *
from lib.sync import Sync

sync = Sync(config)

def infer_helper(request_json):
    with ClearCache():
        with torch.no_grad():
            prep_for_new_gen(sync, request_json, show_info=True)
            
            sync.prep_gen_inputs()
            sync.generate()
            
            make_output_dict_str(sync, show_info=True)
            return sync.dhold.output_dict
    




app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer(manual_request = None):
    if manual_request != None:
        return infer_helper(manual_request)
    else:
        return infer_helper(request.get_json())

if __name__ == '__main__':
    test_mode = True
    models = ["llama3-llava-next-8b", "Meta-Llama-3-70B-Instruct-IQ1_M", "Hermes-2-Theta-Llama-3-8B", "phi-3-vision-128k-instruct"]
    model = models[:]
    
    if not isinstance(model, list):
        model = [model]

    if test_mode:
        for entry in model:
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            infer({'chat': [{'role': 'System', 'content': 'Hello, I am the system.'}, {'role': 'User', 'content': 'hi'}], 'model': entry, 'manual_system_prompt': '', 'use_functions': False, 'model_dtype': 'bfloat16', 'max_new_tokens': '4', 'debugmode': True, 'images': [], 'beam_config': {'use_beam_search': True, 'max_num_beams': '2', 'depth_beams': '3', 'min_conf_for_sure': '0.95', 'min_conf_for_consider': '0.02', 'prob_sum_for_search': '0.98'}})
    else:
        app.run(port=10000)