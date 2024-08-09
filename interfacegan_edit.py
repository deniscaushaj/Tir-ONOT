import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from PIL import Image
import torch
import dnnlib
import legacy
import numpy as np
import argparse
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import subprocess

def is_gpu_free():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    gpu_usage_str = result.stdout.decode('utf-8').strip()
    gpu_usages = gpu_usage_str.split('\n')
    for usage in gpu_usages:
        gpu_usage = int(usage.strip())
        if gpu_usage > 0:
            return False

    return True

while not is_gpu_free():
    print("GPU in use, waiting...")
    time.sleep(1)

print("GPU is free, starting process.")

def free():
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            del obj
    gc.collect()
    
class InterfaceGAN:
    def __init__(self, stylegan_model, boundary, device) -> None:
        self.device = device
        with dnnlib.util.open_url(stylegan_model) as f:
            self.stylegan = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        self.attr_norm = torch.load(boundary)['weight']

    def edit_w(self, wplus, alpha, truncation_psi=1.0):
        with torch.no_grad():
            w_edit = wplus + alpha * self.attr_norm
            img_out = self.stylegan.synthesis(w_edit, noise_mode="const")
            img_out = (img_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        return img_out

free()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylegan", type=str, default="stylegan3-t-ffhq-1024x1024.pkl", help="stylegan model path")
    parser.add_argument("--boundary", type=str, default="resources/interfacegan/boundary_glasses.pth")
    parser.add_argument("--wplus_path", type=str, required=True, help="Path to the saved wplus numpy file")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    wplus = np.load(args.wplus_path)
    wplus = torch.tensor(wplus).to(args.device)

    interfacegan = InterfaceGAN(args.stylegan, args.boundary, args.device)
    outs = []
    for i in tqdm(range(8)):
        edited_out = interfacegan.edit_w(wplus, alpha=0.5 * i)
        outs.append(edited_out)
    outs = np.concatenate(outs, axis=1)
    Image.fromarray(np.uint8(outs)).save("interfacegan_edited.jpg")

free()