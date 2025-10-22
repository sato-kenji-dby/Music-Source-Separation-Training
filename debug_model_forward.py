import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import load_start_checkpoint
import numpy as np

# Build args similar to how inference runs
args = parse_args_inference({'model_type':'htdemucs','config_path':'configs/config_htdemucs_6stems.yaml','start_check_point':'results/HTDemucs4_6 stems.th','device_ids':0})
print('Parsed args')
model, config = get_model_from_config(args.model_type, args.config_path)
print('Created model')
ckpt_path = args.start_check_point
print('Loading checkpoint:', ckpt_path)
try:
    # In newer PyTorch versions torch.load may have weights_only default True.
    # Try explicit weights_only=False to allow custom classes in checkpoints.
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
except TypeError:
    # Older PyTorch may not accept weights_only keyword
    ckpt = torch.load(ckpt_path, map_location='cpu')
except Exception as e:
    print('Failed to load checkpoint:', e)
    raise

print('Checkpoint type/keys:', type(ckpt), list(ckpt.keys()) if isinstance(ckpt, dict) else None)
load_start_checkpoint(args, model, ckpt, type_='inference')
print('Loaded checkpoint into model')

# Move model to cuda if available
device = torch.device('cuda:0' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
model = model.to(device)
model.eval()
print('Using device', device)

# Random input test
channels = getattr(config.training, 'channels', 2)
length = min(131072, int(getattr(config.training, 'samplerate',44100)*2))
print('Random test: channels', channels, 'length', length)
arr = torch.randn(1, channels, length, device=device)
with torch.inference_mode():
    out = model(arr)
print('Random forward output shape:', tuple(out.shape))
try:
    out_np = out.detach().cpu().numpy()
    print('Random out min,max,mean_abs:', float(out_np.min()), float(out_np.max()), float(np.abs(out_np).mean()))
except Exception as e:
    print('Could not convert random output to numpy:', e)

# Now try using first input file chunk
input_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
found_file = None
for root, dirs, files in os.walk(input_root):
    for f in files:
        if f.lower().endswith(('.wav', '.mp3', '.flac')):
            found_file = os.path.join(root, f)
            break
    if found_file:
        break

if not found_file:
    print('No input audio found under', input_root)
    sys.exit(0)

print('Using input file', found_file)
import librosa
mix, sr = librosa.load(found_file, sr=getattr(config.audio,'sample_rate',44100), mono=False)
if mix.ndim == 1:
    mix = np.expand_dims(mix, 0)
print('Loaded mix shape', mix.shape, 'sr', sr)
# take first chunk same as demix chunk size
chunk_size = int(config.training.samplerate * config.training.segment)
chunk = mix[:, :chunk_size]
print('Chunk shape', chunk.shape)
arr2 = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
print('arr2 shape', tuple(arr2.shape))
with torch.inference_mode():
    out2 = model(arr2)
print('Real forward output shape:', tuple(out2.shape))
try:
    out2_np = out2.detach().cpu().numpy()
    print('Real out min,max,mean_abs:', float(out2_np.min()), float(out2_np.max()), float(np.abs(out2_np).mean()))
except Exception as e:
    print('Could not convert real output to numpy:', e)

print('Done')
