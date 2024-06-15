from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
ALLOWED_REFERRERS = ["http://tobs.cloud", "http://www.tobs.cloud", "https://tobs.cloud", "https://www.tobs.cloud"]
with open("allowed_keys.json", "r") as file:
    ALLOWED_API_KEYS = json.loads(file.read())
CORS(app, resources={r"/*": {"origins": ["tobs.cloud", "www.tobs.cloud"]}})  # Allow only the React app


def check_referrer(func):
    def wrapper(*args, **kwargs):
        referrer = request.headers.get("Referer")
        if referrer and any(referrer.startswith(allowed) for allowed in ALLOWED_REFERRERS):
            print(f"access with allowed referrer: {referrer}")
            return func(*args, **kwargs)
        else:
            print(f"not allowed referrer tried to access: {referrer}")
            abort(403)  # Forbidden
    wrapper.__name__ = func.__name__
    return wrapper

def check_api_key(func):
    def wrapper(*args, **kwargs):
        api_key = request.args.get('api_key')
        if api_key in ALLOWED_API_KEYS:
            print(f"access with allowed api_key: {api_key}")
            return func(*args, **kwargs)
        else:
            print(f"not allowed api key tried to access: {api_key}")
            abort(403)  # Forbidden
    wrapper.__name__ = func.__name__
    return wrapper


def index_helper():
    return render_template('index.html')

def get_init_text_helper():
    returned_content = [
        {'role': 'System', 'content': 'Hello, I am the system.'},
    ]
    return json.dumps({'status': 'success', 'returned_content': returned_content})

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
    ai_reply = response.json()['returned_content'][0]
    info = response.json()['info']
    
    returned_content = [
        {'role': 'AI', 'content': "\n".join(ai_reply)}
    ]
    return json.dumps({'status': 'success', 'returned_content': returned_content, 'info': info})



@app.route('/')
@check_api_key
def index():
    return index_helper()

@app.route('/get_init_text')
@check_api_key
def get_init_text():
    return get_init_text_helper()

@app.route('/send_user_msg', methods=['POST'])
@check_api_key
def send_user_msg():
    return send_user_msg_helper(request.get_json())

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=14000, debug=True)