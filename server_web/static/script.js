document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('display-text').innerHTML = "";
    document.getElementById('display-text').chat = [];
    fetchInitText();

    document.getElementById('user-input').addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendUserMsg();
        }
    });
});

function fetchInitText() {
    fetch('/get_init_text')
        .then(response => response.json())
        .then(data => {
            data.returned_content.forEach(item => {
                addText(item);
            });
        });
}

function sendUserMsg() {
    showLoadingAnimation();
    const userInput = document.getElementById('user-input');
    addText({'role': 'User', 'content': userInput.value});
    const chat = document.getElementById('display-text').chat;
    userInput.value = '';
    const model_selector = document.getElementById('modelSelector').value;
    if (model_selector == "default") {
        model_val = "llama3-llava-next-8b";
    } else {
        model_val = model_selector;
    }
    

    fetch('/send_user_msg', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({'chat': chat, 'model': model_val})
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingAnimation();
        console.log(data);
        if ('returned_content' in data) {
            data.returned_content.forEach(item => {
                addText(item);
            });
        }
        if ('info' in data) {
            document.getElementById('info-vram-used').innerHTML = data.info.mem_used;
            document.getElementById('info-vram-total').innerHTML = data.info.mem_total;
            document.getElementById('info-input-tokens').innerHTML = data.info.num_input_tokens;
            document.getElementById('info-output-tokens').innerHTML = data.info.num_output_tokens;
            document.getElementById('info-total-time').innerHTML = data.info.total_time_taken;
            document.getElementById('info-tokens-per-second').innerHTML = data.info.tokens_per_second;
        }
    });
}

function addText(msg_dict) {
    const role = msg_dict.role;
    const content = msg_dict.content;

    document.getElementById('display-text').chat.push(msg_dict);

    let formattedContent = '';
    const maxContentWidth = '30vw';

    if (role === 'System') {
        formattedContent = `<div style="display: flex;"><span style="width: 5vw; min-width: 60px; max-width: 80px; flex-shrink: 0; color: gold; font-weight: bold;">${role}:</span><span style="margin-left: 3vw; flex-shrink: 0; width: ${maxContentWidth}; min-width: 400px;">${content.replace(/\n/g, '<br>')}</span></div>`;
    } else if (role === 'User') {
        formattedContent = `<div style="display: flex;"><span style="width: 5vw; min-width: 60px; max-width: 80px; flex-shrink: 0; color: orange; font-weight: bold; text-shadow: #444 1px 1px 3px;">${role}:</span><span style="margin-left: 3vw; flex-shrink: 0; width: ${maxContentWidth}; min-width: 400px;">${content.replace(/\n/g, '<br>')}</span></div>`;
    } else if (role === 'AI') {
        formattedContent = `<div style="display: flex;"><span style="width: 5vw; min-width: 60px; max-width: 80px; flex-shrink: 0; color: blue; font-weight: bold; text-shadow: #444 1px 1px 3px;">${role}:</span><span style="margin-left: 3vw; flex-shrink: 0; width: ${maxContentWidth}; min-width: 400px;">${content.replace(/\n/g, '<br>')}</span></div>`;
    } else {
        formattedContent = `${content}<br>`;
    }

    document.getElementById('display-text').innerHTML += formattedContent;

    if (document.getElementById('display-text').scrollHeight > document.getElementById('display-text').clientHeight) {
        document.getElementById('display-text').scrollTop = document.getElementById('display-text').scrollHeight;
    }
}

function showLoadingAnimation() {
    document.getElementById('loading-animation').style.removeProperty("display");
}

function hideLoadingAnimation() {
    document.getElementById('loading-animation').style.display = "none";
}

function clearChat() {
    console.log("clicked clearChat button");
    document.getElementById('display-text').innerHTML = "";
    document.getElementById('display-text').chat = [];
    fetchInitText();
}

function changeModelOpen() {
    console.log("changed model");
}

function moreInfoCheckbox() {
    console.log("changed moreinfo checkbox");
}