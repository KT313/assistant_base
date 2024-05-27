from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)




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
    ai_reply = response.json()['returned_content']
    info = response.json()['info']
    
    returned_content = [
        {'role': 'AI', 'content': "\n".join(ai_reply)}
    ]
    return json.dumps({'status': 'success', 'returned_content': returned_content, 'info': info})



@app.route('/')
def index():
    return index_helper()

@app.route('/get_init_text')
def get_init_text():
    return get_init_text_helper()

@app.route('/send_user_msg', methods=['POST'])
def send_user_msg():
    return send_user_msg_helper(request.get_json())

if __name__ == '__main__':
    app.run(port=14000, debug=True)