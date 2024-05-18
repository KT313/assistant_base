from flask import Flask, render_template, request, jsonify
import requests
import weave
import json

app = Flask(__name__)


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



@weave.op()
def index_helper():
    return render_template('index.html')

@weave.op()
def get_init_text_helper():
    returned_content = [
        {'role': 'System', 'content': 'Hello, I am the system.'},
    ]
    return json.dumps({'status': 'success', 'returned_content': returned_content})

@weave.op()
def send_user_msg_helper(request_json):
    data = request_json
    chat = data['chat']

    url = f"http://127.0.0.1:10000/infer"
    data_to_send = request_json
    data_to_send['use_image'] = False
    response = requests.post(url, json=data_to_send)   
    if response.json()['status'] == "error":
        print(response.json()['error-info'])
        return json.dumps(response.json())
    ai_reply = response.json()['returned_content']
    info = response.json()['info']
    
    returned_content = [
        {'role': 'AI', 'content': "\n".join(ai_reply)}
    ]
    return json.dumps({'status': 'success', 'returned_content': returned_content, 'info': info})



@app.route('/')
def index():
    init_client = InitializedClient(client)
    return index_helper()

@app.route('/get_init_text')
def get_init_text():
    init_client = InitializedClient(client)
    return get_init_text_helper()

@app.route('/send_user_msg', methods=['POST'])
def send_user_msg():
    init_client = InitializedClient(client)
    return send_user_msg_helper(request.get_json())

if __name__ == '__main__':
    app.run(port=14000, debug=True)