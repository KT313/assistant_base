from lib.misc import *
from lib.imports import *
from lib.sync import Sync

sync = Sync(config)

def infer_helper(request_json):
    with ClearCache():
        with torch.no_grad():
            prep_for_new_gen(sync, request_json, show_info=False)
            
            sync.prep_gen_inputs()
            sync.generate()
            
            make_output_dict_str(sync, show_info=False)
            return sync.dhold.output_dict
    




app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    return infer_helper(request.get_json())

if __name__ == '__main__':
    app.run(port=10000)