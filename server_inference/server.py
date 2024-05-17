from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import torch
import copy
import json

pretrained = "../../../models/multimodal/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

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



@weave.op()
def infer_helper(request_json):
    data = request_json
    chat = data['chat']
    if not isinstance(chat, list):
        chat = [chat]
    use_image = data['use_image']

    # make conversation excluding last message. System messages are excluded.
    conv_template = "llava_llama_3"
    conv = copy.deepcopy(conv_templates[conv_template])
    for entry in chat[:-1]:
        if entry['role'] == "User":
            conv.append_message(conv.roles[0], entry['content'])
        elif entry['role'] == "AI":
            conv.append_message(conv.roles[1], entry['content'])

    # with image
    if use_image:
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        
        question = DEFAULT_IMAGE_TOKEN + f"\n{chat[-1]['content']}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        print(f"\n\n\nprompt:\n{prompt_question}\n\n\n")
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
    else:
        question = chat[-1]['content']
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        print(f"\n\n\nprompt:\n{prompt_question}\n\n\n")
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
        cont = model.generate(
            input_ids,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    returned_content = [entry.strip() for entry in text_outputs]
    return json.dumps({'status': 'success', 'returned_content': returned_content})



from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    init_client = InitializedClient(client)
    return infer_helper(request.get_json())

if __name__ == '__main__':
    app.run(port=10000)