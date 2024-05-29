import warnings
import requests
import base64
import time
import math
import copy
import json
import os
import io
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, AutoProcessor, PaliGemmaForConditionalGeneration
from flask import Flask, render_template, request, jsonify
from scipy.special import softmax
from llama_cpp import Llama
from tqdm import tqdm
from PIL import Image
import bitsandbytes
import numpy as np
import flash_attn
import torch


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from conversation import conv_templates, SeparatorStyle







