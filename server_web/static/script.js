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
    const manualSystemPrompt = document.getElementById('manual-system-prompt');
    const max_new_tokens = document.getElementById('max_new_tokens');

    const debugmode = document.getElementById('debugmode-checkbox');
    
    const use_beam_search = document.getElementById('usebeamsearch-checkbox');
    const max_num_beams = document.getElementById('max_num_beams');
    const depth_beams = document.getElementById('depth_beams');
    const min_conf_for_sure = document.getElementById('min_conf_for_sure');
    const min_conf_for_consider = document.getElementById('min_conf_for_consider');
    const prob_sum_for_search = document.getElementById('prob_sum_for_search');
    
    const usefunctions = document.getElementById('usefunctions-checkbox');
    
    const modeldtype = document.querySelector('input[name="model-dtype"]:checked');
    
    const model_selector = document.getElementById('modelSelector').value;
    if (model_selector == "default") {
        model_val = "llama3-llava-next-8b";
    } else {
        model_val = model_selector;
    }

    // Get the uploaded images as Base64-encoded strings
    const imageGrid = document.querySelector('.image-grid');
    const imageElements = imageGrid.querySelectorAll('div');
    const base64Images = Array.from(imageElements).map(element => {
        const backgroundImage = element.style.backgroundImage;
        const base64Data = backgroundImage.slice(5, -2); // Remove the 'url("' and '")' parts
        return base64Data;
    });
    

    fetch('/send_user_msg', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'chat': chat, 
            'model': model_val, 
            'manual_system_prompt': manualSystemPrompt.value, 
            'use_functions': usefunctions.checked, 
            'model_dtype': modeldtype.value, 
            'max_new_tokens': max_new_tokens.value, 
            'debugmode': debugmode.checked, 
            'beam_config': {
                'use_beam_search': use_beam_search.checked, 
                'max_num_beams': max_num_beams.value, 
                'depth_beams': depth_beams.value, 
                'min_conf_for_sure': min_conf_for_sure.value, 
                'min_conf_for_consider': min_conf_for_consider.value, 
                'prob_sum_for_search': prob_sum_for_search.value
            },
            'images': base64Images
        })
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
    var content = msg_dict.content;

    content = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    content = content.replace(/\n/g, '<br>');

    console.log(content);

    document.getElementById('display-text').chat.push(msg_dict);

    let formattedContent = '';
    const maxContentWidth = '30vw';

    if (role === 'System') {
        formattedContent = `<div style="display: flex; white-space: pre-wrap; overflow: auto;"><span style="width: 5vw; min-width: 60px; max-width: 80px; flex-shrink: 0; color: gold; font-weight: bold;">${role}:</span><span style="margin-left: 3vw; flex-shrink: 0; width: ${maxContentWidth}; min-width: 400px;">${content}</span></div>`;
    } else if (role === 'User') {
        formattedContent = `<div style="display: flex; white-space: pre-wrap; overflow: auto;"><span style="width: 5vw; min-width: 60px; max-width: 80px; flex-shrink: 0; color: orange; font-weight: bold; text-shadow: #444 1px 1px 3px;">${role}:</span><span style="margin-left: 3vw; flex-shrink: 0; width: ${maxContentWidth}; min-width: 400px;">${content}</span></div>`;
    } else if (role === 'AI') {
        formattedContent = `<div style="display: flex; white-space: pre-wrap; overflow: auto;"><span style="width: 5vw; min-width: 60px; max-width: 80px; flex-shrink: 0; color: blue; font-weight: bold; text-shadow: #444 1px 1px 3px;">${role}:</span><span style="margin-left: 3vw; flex-shrink: 0; width: ${maxContentWidth}; min-width: 400px;">${content}</span></div>`;
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
    hideLoadingAnimation();
    fetchInitText();
}

function changeModelOpen() {
    console.log("changed model");
}

function moreInfoCheckbox() {
    console.log("changed moreinfo checkbox");
}

function usefunctionsCheckbox() {
    console.log("changed usefunctions checkbox");
}

function usebeamsearchCheckbox() {
    console.log("changed usebeamsearch checkbox");
}

function debugmodeCheckbox() {
    console.log("changed debugmode checkbox");
}

function upload() {
    const fileUploadInput = document.querySelector('.file-uploader');
    const files = Array.from(fileUploadInput.files);

    // Check if any files are selected
    if (files.length === 0) {
        return;
    }

    // Check if all selected files are images
    const allImagesValid = files.every(file => file.type.includes('image'));
    if (!allImagesValid) {
        return alert('Only images are allowed!');
    }

    // Check if total size exceeds 10 MB
    const totalSize = files.reduce((total, file) => total + file.size, 0);
    if (totalSize > 10_000_000) {
        return alert('Maximum upload size is 10MB!');
    }

    const imageGrid = document.querySelector('.image-grid');
    imageGrid.innerHTML = ''; // Clear previous content

    const gridSize = getGridSize(files.length);
    const imageSize = `calc(100% / ${gridSize})`;

    files.forEach(file => {
        const fileReader = new FileReader();
        fileReader.readAsDataURL(file);

        fileReader.onload = (fileReaderEvent) => {
            const imageElement = document.createElement('div');
            imageElement.style.backgroundImage = `url(${fileReaderEvent.target.result})`;
            imageElement.style.width = imageSize;
            imageElement.style.height = imageSize;
            imageElement.style.backgroundSize = 'cover';
            imageElement.style.backgroundPosition = 'center';
            imageGrid.appendChild(imageElement);
        };
    });
}

function getGridSize(numImages) {
    if (numImages <= 1) {
        return 1;
    } else if (numImages <= 4) {
        return 2;
    } else if (numImages <= 16) {
        return 4;
    } else {
        return Math.ceil(Math.sqrt(numImages));
    }
}

function clearImages() {
    const imageGrid = document.querySelector('.image-grid');
    imageGrid.innerHTML = ''; // Clear previous content
}