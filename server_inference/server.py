from lib.misc import *
from lib.imports import *
from lib.sync import Sync

sync = Sync(config)

def infer_helper(request_json):
    if sync.mode != "assistant":
        sync.change_mode("assistant")
    
    with ClearCache():
        with torch.no_grad():
            prep_for_new_gen(sync, request_json, show_info=True)
            
            if sync.dhold.inputs['agent_task_mode']: sync.solve_agent_task()
            else: sync.generate()
            
            make_output_dict_str(sync, show_info=False)
            print("\nfinal output:")
            for key, val in json.loads(sync.dhold.output_dict).items():
                print(" |", f"{key}: {val}")
            return sync.dhold.output_dict


def image_gen_helper(request_json):
    if sync.mode != "image_gen":
        sync.change_mode("image_gen")
        
    with ClearCache():
        with torch.no_grad():
            prep_for_new_image_gen(sync, request_json, show_info=True)
            sync.generate_image()
            make_output_dict_image_gen_str(sync, show_info=False)
            return sync.dhold.output_dict
    




app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer(manual_request = None):
    request_dict = request.get_json() if manual_request == None else manual_request
    
    if "model_dtype" not in request_dict: request_dict['model_dtype'] = "bfloat16"
    if "max_new_tokens" not in request_dict: request_dict['max_new_tokens'] = "512"
    if "debugmode" not in request_dict: request_dict['debugmode'] = False
    if "images" not in request_dict: request_dict['images'] = []
    if "agent_task_mode" not in request_dict: request_dict['agent_task_mode'] = False
    if "use_functions" not in request_dict: request_dict['use_functions'] = False
    if "use_voiceinput" not in request_dict: request_dict['use_voiceinput'] = False
    if "use_voiceoutput" not in request_dict: request_dict['use_voiceoutput'] = False
    if "use_voiceoutput" not in request_dict: request_dict['use_voiceoutput'] = False
    if "beam_config" not in request_dict: request_dict['beam_config'] = {
        "use_beam_search": False,
        "max_num_beams": "1",
        "depth_beams": "1",
        "min_conf_for_sure": "0.95",
        "min_conf_for_consider": "0.02",
        "prob_sum_for_search": "0.98",
    }
        
    return infer_helper(request_dict)

@app.route('/image_gen', methods=['POST'])
def image_gen(manual_request = None):
    if manual_request != None:
        return image_gen_helper(manual_request)
    else:
        return image_gen_helper(request.get_json())


if __name__ == '__main__':
    test_mode = False
    multi_turn = False
    models = ["test", "llama3-llava-next-8b", "Meta-Llama-3-70B-Instruct-IQ1_M", "Hermes-2-Theta-Llama-3-8B", "phi-3-vision-128k-instruct"]
    models_a = ["Hermes-2-Theta-Llama-3-8B", "Meta-Llama-3-70B-Instruct-IQ1_M", "test"]
    model = models_a[:]

    if test_mode: test_models(model, test_mode, multi_turn, infer)
    else: app.run(port=10000)