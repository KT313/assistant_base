function getApiKeyFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('api_key');
}

function isLocalhost() {
  const hostname = window.location.hostname;
  return hostname === 'localhost' || hostname === '127.0.0.1';
}


document.addEventListener('DOMContentLoaded', function() {
    const apiKey = getApiKeyFromUrl();
    document.apiKey = apiKey;
    if (document.apiKey) {
        console.log('API Key:', document.apiKey);
        // Use the API key for further operations
        // Example: make an API call
        // fetch(`http://example.com/api?api_key=${apiKey}`)
    } else {
        console.error('API key not found in the URL');
    }
    //console.log("document.api_key:", document.api_key);
    const tabs = document.querySelectorAll('nav ul li a');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();

            tabs.forEach(tab => tab.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            this.classList.add('active');
            document.getElementById(this.getAttribute('data-tab')).classList.add('active');
        });
    });
    
    document.getElementById('display-text').innerHTML = "";
    document.getElementById('display-text').chat = [];
    fetchInitText();

    document.getElementById('user-input').addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendUserMsg();
        }
    });

    // image generation
    document.getElementById('image-generation-user-input').addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendUserMsgImageGen();
        }
    });
    
});

function fetchInitText() {
    var url = `http://${window.location.hostname}:14000/get_init_text`
    try {
        if (document.apiKey) {url = url + `?api_key=${document.apiKey}`;}
    } catch {}
    if (isLocalhost()) {
        url = 'http://127.0.0.1:14000/get_init_text'
    }
    fetch(url)
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
    const usevoiceinput = document.getElementById('usevoiceinput-checkbox');
    const usevoiceoutput = document.getElementById('usevoiceoutput-checkbox');
    const allowimagegen = document.getElementById('allowimagegen-checkbox');
    const agenttaskmode = document.getElementById('agenttaskmode-checkbox');
    
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
    
    var url = `http://${window.location.hostname}:14000/send_user_msg`
    try {
        if (document.apiKey) {url = url + `?api_key=${document.apiKey}`;}
    } catch {}
    if (isLocalhost()) {
        url = 'http://127.0.0.1:14000/send_user_msg'
    }
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'chat': chat, 
            'model': model_val, 
            'manual_system_prompt': manualSystemPrompt.value, 
            'use_functions': usefunctions.checked, 
            'use_voiceinput': usevoiceinput.checked, 
            'use_voiceoutput': usevoiceoutput.checked, 
            'allow_imagegen': allowimagegen.checked, 
            'agent_task_mode': agenttaskmode.checked, 
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
        if (data['status'] == "error") {
            var currentdate = new Date(); 
            hour = String(currentdate.getHours()).padStart(2, '0');
            minutes = String(currentdate.getMinutes()).padStart(2, '0');
            seconds = String(currentdate.getSeconds()).padStart(2, '0');
            formattedContent = `<div style="display: flex; white-space: pre-wrap; overflow: auto;"><span style="flex-shrink: 0;">${hour+':'+minutes+':'+seconds+' '+data['error-info']}</span></div>`;
            document.getElementById('log-content').innerHTML = formattedContent + document.getElementById('log-content').innerHTML;
        } else {
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

function showLoadingAnimationImageGen() {
    document.getElementById('image-generation-loading-animation').style.removeProperty("display");
}

function hideLoadingAnimationImageGen() {
    document.getElementById('image-generation-loading-animation').style.display = "none";
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

function usevoiceinputCheckbox() {
    console.log("changed usevoiceinput checkbox");
}

function usevoiceoutputCheckbox() {
    console.log("changed usevoiceoutput checkbox");
}

function allowimagegenCheckbox() {
    console.log("changed allowimagegen checkbox");
}

function agenttaskmodeCheckbox() {
    console.log("changed agenttaskmode checkbox");
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

function sendUserMsgImageGen() {
    showLoadingAnimationImageGen();
    const userInput = document.getElementById('image-generation-user-input').value;
    const userInputNeg = document.getElementById('image-generation-user-input-negative').value;
    
    // addText({'role': 'User', 'content': userInput.value});
    //const chat = document.getElementById('display-text').chat;
    // userInput.value = '';

    const debugmode = document.getElementById('image-generation-debugmode-checkbox');
    
    const model_selector = document.getElementById('image-generation-modelSelector').value;
    const sampler_selector = document.getElementById('samplerSelector').value;
    const cfg_selector = document.getElementById('cfg').value;
    const steps_selector = document.getElementById('steps').value;
    const clip_skip_selector = document.getElementById('clip-skip').value;
    const image_gen_batch_size_selector = document.getElementById('image_gen_batch_size').value;
    const image_gen_width_selector = document.getElementById('image_gen_width').value;
    const image_gen_height_selector = document.getElementById('image_gen_height').value;
    const seed_selector = document.getElementById('seed').value;
    
    var url = `http://${window.location.hostname}:14000/send_user_msg_image_gen`
    try {
        if (document.apiKey) {url = url + `?api_key=${document.apiKey}`;}
    } catch {}
    if (isLocalhost()) {
        url = 'http://127.0.0.1:14000/send_user_msg_image_gen'
    }
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'model': model_selector, 

            'prompt': userInput,
            'prompt_neg': userInputNeg,
            'sampler': sampler_selector,
            'cfg': cfg_selector,
            'steps': steps_selector,
            'clip_skip': clip_skip_selector,
            'batch_size': image_gen_batch_size_selector,
            'width': image_gen_width_selector,
            'height': image_gen_height_selector,
            'seed': seed_selector,
            
            'debugmode': debugmode.checked, 
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingAnimationImageGen();
        console.log(data);
        if (data['status'] === "error") {
            var currentdate = new Date(); 
            var hour = String(currentdate.getHours()).padStart(2, '0');
            var minutes = String(currentdate.getMinutes()).padStart(2, '0');
            var seconds = String(currentdate.getSeconds()).padStart(2, '0');
            var formattedContent = `<div style="display: flex; white-space: pre-wrap; overflow: auto;"><span style="flex-shrink: 0;">${hour + ':' + minutes + ':' + seconds + ' ' + data['error-info']}</span></div>`;
            document.getElementById('image-log-content').innerHTML = formattedContent + document.getElementById('image-log-content').innerHTML;
        } else {
            if ('generated_image' in data) {
                console.log("should show image now");
                const imageGrid = document.getElementById('image-generation-output');
                imageGrid.innerHTML = ''; // Clear previous content
    
                const files = data['generated_image'];
                const gridSize = getGridSize(files.length);
                const imageSize = `calc(100% / ${gridSize})`;
                
                files.forEach(file => {
                    const imageElement = document.createElement('div');
                    imageElement.style.backgroundImage = `url(data:image/png;base64,${file})`;
                    imageElement.style.width = imageSize;
                    imageElement.style.height = imageSize;
                    imageElement.style.backgroundSize = 'cover';
                    imageElement.style.backgroundPosition = 'center';
                    imageElement.style.display = 'inline-block';
                    imageElement.style.boxSizing = 'border-box';
                    imageGrid.appendChild(imageElement);
                });
            }
            if ('info' in data) {
                document.getElementById('image-info-vram-used').innerHTML = data.info.mem_used;
                document.getElementById('image-info-vram-total').innerHTML = data.info.mem_total;
                document.getElementById('image-info-iterations-per-second').innerHTML = data.info.iterations_per_second;
                document.getElementById('image-info-images-per-second').innerHTML = data.info.images_per_second;
                document.getElementById('image-info-total-time').innerHTML = data.info.total_time_taken;
                /*
                document.getElementById('info-vram-used').innerHTML = data.info.mem_used;
                document.getElementById('info-vram-total').innerHTML = data.info.mem_total;
                document.getElementById('info-input-tokens').innerHTML = data.info.num_input_tokens;
                document.getElementById('info-output-tokens').innerHTML = data.info.num_output_tokens;
                document.getElementById('info-total-time').innerHTML = data.info.total_time_taken;
                document.getElementById('info-tokens-per-second').innerHTML = data.info.tokens_per_second;
                */
            }
        }
    });
}