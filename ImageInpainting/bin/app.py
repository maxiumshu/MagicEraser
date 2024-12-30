#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
import base64
from io import BytesIO
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

app = Flask(__name__)
CORS(app)

model = None
train_config = None

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def load_model(predict_config: OmegaConf):
    global model
    try:
        if sys.platform != 'win32':
            'register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log'

        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path, 
                                        'models', 
                                        predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if model is None or train_config is None:
            raise ValueError("Model is None after loading.")

    except Exception as e:
        logger.critical(f"Failed to load model because of: {str(e)}.")
        sys.exit(1)


load_model()


def padding(base64URL):
    missing_padding = 4 - len(base64URL) % 4
    if missing_padding:
        base64URL += '=' * missing_padding
    return base64URL


def base64_to_image(base64_str):
    base64_str = padding(base64_str)

    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]

    img_data = base64.urlsafe_b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img


def load_image_from_base64(base64_str):
    img = base64_to_image(base64_str).convert('RGB')
    img = np.array(img)
    
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    
    out_img = img.astype('float32') / 255
    return out_img


def load_mask_from_base64(base64_str):
    mask = base64_to_image(base64_str).convert('L')
    mask = np.array(mask)
    
    return mask[None, ...] 


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def process_base64_data(image_base64, mask_base64, pad_out_to_modulo=None):
    image = load_image_from_base64(image_base64)
    mask = load_mask_from_base64(mask_base64)

    result = dict(image=image, mask=mask)
    
    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        result['unpad_to_size'] = result['image'].shape[1:] 
        result['image'] = pad_img_to_modulo(result['image'], pad_out_to_modulo)
        result['mask'] = pad_img_to_modulo(result['mask'], pad_out_to_modulo)

    return result


def extract_format(base64URL):
    if base64URL.startswith("data"):
        mime_type = base64URL.split(";")[0].split(":")[1]
        image_format = mime_type.split("/")[1]
        return image_format
    else:
        return None


def process_Img(image_data, mask_data):
    try:
        with open("/home/experiment7/MagicEraser/ImageInpainting/configs/prediction/default.yaml", 'r') as f:
            predict_config = yaml.safe_load(f)

        image_format = extract_format(image_data)

        device = torch.device("cpu")
        batch = process_base64_data(image_data, mask_data, predict_config['dataset']['pad_out_to_modulo'])
        batch = default_collate([batch])
       
        if predict_config.get('refine') is True:
            assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
            result = refine_predict(batch, model, **predict_config.get('refiner'))
            result = result[0].permute(1,2,0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                batch = move_to_device(batch, device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                result = batch[predict_config.get('out_key')][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size')
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    result = result[:orig_height, :orig_width]

        result = np.clip(result * 255, 0, 255).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(f'.{image_format}', result)
        resultURL = base64.b64encode(buffer)
        resultURL = resultURL.decode()
        return f"data:image/{image_format};base64,{resultURL}"
    except Exception as e :
        logger.critical(f"Error during image processing: {str(e)}")
        return None

# Flask app router
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_file = data.get('image')
        mask_file = data.get('mask')

        resultURL = process_Img(image_file, mask_file)
        # resultURL = "Test"
        return jsonify({'result_image': resultURL})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=7777)
