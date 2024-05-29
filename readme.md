## Web assistant (WIP)  

- image support  
- multiple models  
- custom adaptive beam search  

note: custom adaptive beam search is slow with llama-cpp models (quantized LLama-3-70B) and Llama-3-Llava (only when using an image) right now because they have to do the beam searches in sequence, couldn't get them to work in batch. For llama-cpp it's because it (for whatever reason???) does not support batch inference. Llama-3-Llava does support it but when you supply an image additionally it gets confused (and so do I).

![Web UI](misc/web_ui.png)