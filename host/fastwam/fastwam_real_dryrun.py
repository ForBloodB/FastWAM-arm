import argparse
import io
import json
import os
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image
import torch

from fastwam_b601_adapter import adapt_libero_action_to_b601
from fastwam_inference_smoke import _load_policy

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/home/aup/lerobot-fastwam'))
parser.add_argument('--state', type=Path, required=True)
parser.add_argument('--context', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--steps', type=int, default=1)
args = parser.parse_args()
root = args.root.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)

def get(path):
    with urllib.request.urlopen('http://127.0.0.1:8765' + path, timeout=5) as response:
        return response.read()

status = json.loads(get('/api/status'))
if not status.get('live') or not status.get('side_camera', {}).get('live'):
    raise RuntimeError('Both real cameras must be live')
started_ns = time.monotonic_ns()
external_bgr = cv2.imdecode(np.frombuffer(get('/side/frame.jpg'), np.uint8), cv2.IMREAD_COLOR)
wrist_bgr = cv2.imdecode(np.frombuffer(get('/frame.jpg'), np.uint8), cv2.IMREAD_COLOR)
ended_ns = time.monotonic_ns()
if external_bgr is None or wrist_bgr is None:
    raise RuntimeError('Real camera JPEG decode failed')
external = cv2.cvtColor(cv2.resize(external_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
wrist = cv2.cvtColor(cv2.resize(wrist_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
Image.fromarray(external).save(out / 'external.png')
Image.fromarray(wrist).save(out / 'wrist.png')
rgb = np.concatenate((external, wrist), axis=1)
Image.fromarray(rgb).save(out / 'combined.png')
state_document = json.loads(args.state.read_text())
raw_state = np.asarray(state_document['raw_state'], dtype=np.float32)
if raw_state.shape != (8,) or not np.isfinite(raw_state).all():
    raise ValueError('Invalid real robot state')

os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(root / 'checkpoints')
os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
checkpoint = root / 'checkpoints/fastwam_release/libero_optional_idm_2cam224.pt'
stats = root / 'checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json'
cfg, model, processor = _load_policy(root, checkpoint, stats, args.steps)
context_blob = torch.load(args.context, map_location='cpu', weights_only=True)
state_key = processor.shape_meta['state'][0]['key']
state_batch = processor.action_state_transform({'state': {state_key: torch.from_numpy(raw_state).unsqueeze(0)}})
normalized_state = processor.normalizer.forward(state_batch)['state'][state_key]
image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device='cuda', dtype=model.torch_dtype)
image = image * (2.0 / 255.0) - 1.0
started = time.perf_counter()
with torch.inference_mode():
    result = model.infer_action(
        prompt=None, input_image=image, action_horizon=4,
        context=context_blob['context'].to('cuda'), context_mask=context_blob['context_mask'].to('cuda'),
        proprio=normalized_state.to('cuda'), num_inference_steps=args.steps,
        seed=1234, rand_device='cpu', compile_action_infer=False,
        action_infer_mode='first_frame', negative_prompt='', text_cfg_scale=1.0, tiled=False)
inference_s = time.perf_counter() - started
normalized_action = result['action'].detach().float().cpu()
action_key = processor.shape_meta['action'][0]['key']
normalizer = processor.normalizer.normalizers['action'][action_key]
libero_action = normalizer.backward(normalized_action).numpy()
if libero_action.shape != (4, 7) or not np.isfinite(libero_action).all():
    raise RuntimeError(f'Invalid model action {libero_action.shape}')
adapted = adapt_libero_action_to_b601(raw_state, libero_action[0],
    max_translation_step_m=.005, max_rotation_step_rad=np.deg2rad(1.0))
output = {
    'result': 'real_inference_only_no_dispatch',
    'checkpoint': str(checkpoint),
    'checkpoint_embodiment': 'Panda/LIBERO zero-shot weights; not B601 fine-tuned',
    'prompt': context_blob.get('prompt'),
    'device': torch.cuda.get_device_name(0),
    'inference_seconds': inference_s,
    'camera_source': 'live 395 side USB and wrist Gemini 305 HTTP frames',
    'camera_capture_elapsed_ms': (ended_ns - started_ns) / 1e6,
    'robot_state': state_document,
    'normalized_action': normalized_action.tolist(),
    'libero_action_after_checkpoint_stats': libero_action.tolist(),
    'b601_adapter_bounded_to_5mm_1deg': adapted,
    'cross_embodiment_mapping_validated': False,
    'hardware_execution_enabled': False,
}
(out / 'inference.json').write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
