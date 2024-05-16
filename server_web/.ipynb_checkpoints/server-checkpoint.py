from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_init_text')
def get_init_text():
    returned_content = [
        {'role': 'System', 'content': 'Hello, I am the system.'},
    ]
    return jsonify({'status': 'success', 'returned_content': returned_content})

@app.route('/send_user_msg', methods=['POST'])
def send_user_msg():
    data = request.get_json()
    chat = data['chat']

    url = f"http://127.0.0.1:10000/infer"
    data_to_send = {}
    data_to_send['chat'] = chat
    data_to_send['use_image'] = False
    response = requests.post(url, json=data_to_send)    
    ai_reply = response.json()['returned_content']
    
    returned_content = [
        {'role': 'AI', 'content': "\n".join(ai_reply)}
    ]
    return jsonify({'status': 'success', 'returned_content': returned_content})

if __name__ == '__main__':
    app.run(port=14000)