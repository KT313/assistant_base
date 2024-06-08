from lib.misc import *
from lib.imports import *
from lib.sync import Sync

sync = Sync(config)

def infer_helper(request_json):
    with ClearCache():
        with torch.no_grad():
            prep_for_new_gen(sync, request_json, show_info=True)
            
            if sync.dhold.inputs['agent_task_mode']: sync.solve_agent_task()
            else: sync.generate()
            
            make_output_dict_str(sync, show_info=False)
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
    multi_turn = False
    models = ["llama3-llava-next-8b", "Meta-Llama-3-70B-Instruct-IQ1_M", "Hermes-2-Theta-Llama-3-8B", "phi-3-vision-128k-instruct"]
    model = models[2]

    if test_mode: test_models(model, test_mode, multi_turn, infer)
    else: app.run(port=10000)