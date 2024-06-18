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
    if manual_request != None:
        return infer_helper(manual_request)
    else:
        return infer_helper(request.get_json())

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