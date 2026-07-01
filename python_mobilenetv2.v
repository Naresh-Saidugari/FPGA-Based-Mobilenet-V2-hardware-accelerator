"""
================================================================================
MOBILENETV2 FPGA INFERENCE - WITH FIXED WEIGHTS FOR PYNQ-ZU
================================================================================
"""

import numpy as np
from pynq import Overlay
from pynq import allocate
import os
from PIL import Image
import time

BITSTREAM_PATH = "./designn.bit"
WEIGHT_DIR = "/tmp/weights_fixed5"
IMAGE_PATH = "./app.jpg"

# =====================================================================
# HARDWARE MAC (YOUR VERILOG MAC IN PL)
# =====================================================================
class HardwareMAC3x3:
    def __init__(self, bitstream_path):
        print(f"[HW] Loading bitstream: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)
        self.dma = self.overlay.axi_dma_0
        
        self.in_buffer = allocate(shape=(3,), dtype=np.uint64)
        self.out_buffer = allocate(shape=(1,), dtype=np.int32)
        print("[HW] Single-MAC DMA ready")
        
    def _pack(self, x_vals, y_vals):
        x = np.asarray(x_vals, dtype=np.int8).flatten()
        y = np.asarray(y_vals, dtype=np.int8).flatten()
        assert len(x) == 9 and len(y) == 9
        
        word0 = sum((int(x[i]) & 0xFF) << (8 * i) for i in range(8))
        word1 = (int(x[8]) & 0xFF)
        word1 |= sum((int(y[i]) & 0xFF) << (8 * (i + 1)) for i in range(7))
        word2 = (int(y[7]) & 0xFF) | ((int(y[8]) & 0xFF) << 8)
        
        return word0, word1, word2
        
    def run_mac(self, inputs, weights):
        w0, w1, w2 = self._pack(inputs, weights)
        self.in_buffer[0] = w0
        self.in_buffer[1] = w1
        self.in_buffer[2] = w2
        self.in_buffer.flush()
        
        self.dma.recvchannel.transfer(self.out_buffer)
        self.dma.sendchannel.transfer(self.in_buffer)
        
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        
        self.out_buffer.invalidate()
        return float(self.out_buffer[0])
    
    def cleanup(self):
        self.in_buffer.freebuffer()
        self.out_buffer.freebuffer()
        print("[HW] Buffers freed")

# =====================================================================
# FPGA CONV ENGINE
# =====================================================================
class Conv3x3_HW:
    def __init__(self, hw_mac):
        self.hw = hw_mac
        
    def _quantize(self, ifm):
        max_val = np.max(np.abs(ifm))
        scale = 127.0 / max_val if max_val > 1e-8 else 1.0
        q = np.clip(np.round(ifm * scale), -128, 127).astype(np.int8)
        return q, scale

    def conv2d(self, ifm, weights, weight_scale, stride=1, padding=1):
        out_ch, in_ch = weights.shape[0], weights.shape[1]
        padded = np.pad(ifm, ((0,0),(padding,padding),(padding,padding)), mode='constant') if padding else ifm
        out_h = (padded.shape[1] - 3) // stride + 1
        out_w = (padded.shape[2] - 3) // stride + 1
        ifm_q, input_scale = self._quantize(padded)
        out = np.zeros((out_ch, out_h, out_w), dtype=np.float32)
        deq = weight_scale * input_scale
        
        w_flat = {}
        for co in range(out_ch):
            for ci in range(in_ch):
                w_flat[(co, ci)] = weights[co, ci].flatten().tolist()
        
        for co in range(out_ch):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh*stride, ow*stride
                    acc = 0.0
                    for ci in range(in_ch):
                        acc += self.hw.run_mac(
                            ifm_q[ci, h0:h0+3, w0:w0+3].flatten().tolist(),
                            w_flat[(co, ci)]
                        )
                    out[co, oh, ow] = acc / deq
        return out

    def depthwise_conv2d(self, ifm, weights, weight_scale, stride=1, padding=1):
        ch = weights.shape[0]
        padded = np.pad(ifm, ((0,0),(padding,padding),(padding,padding)), mode='constant') if padding else ifm
        out_h = (padded.shape[1] - 3) // stride + 1
        out_w = (padded.shape[2] - 3) // stride + 1
        ifm_q, input_scale = self._quantize(padded)
        out = np.zeros((ch, out_h, out_w), dtype=np.float32)
        deq = weight_scale * input_scale
        
        w_flat = {}
        for c in range(ch):
            w_flat[c] = weights[c, 0].flatten().tolist()
        
        for c in range(ch):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh*stride, ow*stride
                    out[c,oh,ow] = self.hw.run_mac(
                        ifm_q[c, h0:h0+3, w0:w0+3].flatten().tolist(),
                        w_flat[c]
                    ) / deq
        return out

    def conv1x1(self, ifm, weights, weight_scale, stride=1):
        out_ch, in_ch = weights.shape[0], weights.shape[1]
        _, h, w = ifm.shape
        ifm_q, input_scale = self._quantize(ifm)
        out = np.zeros((out_ch, h, w), dtype=np.float32)
        deq = weight_scale * input_scale
        
        w_flat = {}
        for co in range(out_ch):
            for chunk in range(0, in_ch, 9):
                wv = weights[co, chunk:chunk+9, 0, 0]
                if len(wv) < 9:
                    wv = np.pad(wv, (0, 9-len(wv)), 'constant')
                w_flat[(co, chunk)] = wv.flatten().tolist()
        
        for co in range(out_ch):
            for i in range(h):
                for j in range(w):
                    acc = 0.0
                    for chunk in range(0, in_ch, 9):
                        xv = ifm_q[chunk:chunk+9, i, j]
                        if len(xv) < 9:
                            xv = np.pad(xv, (0, 9-len(xv)), 'constant')
                        acc += self.hw.run_mac(xv.flatten().tolist(), w_flat[(co, chunk)])
                    out[co, i, j] = acc / deq
        return out

# =====================================================================
# MOBILENETV2 BLOCKS
# =====================================================================
def relu6(x): return np.clip(x, 0.0, 6.0)

def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    g = gamma.astype(np.float32).reshape(-1,1,1)
    b = beta.astype(np.float32).reshape(-1,1,1)
    m = mean.astype(np.float32).reshape(-1,1,1)
    v = var.astype(np.float32).reshape(-1,1,1)
    return g * ((x - m) / np.sqrt(v + eps)) + b

class InvertedResidualBlock:
    def __init__(self, conv_hw, load_w, load_bn, idx):
        self.conv = conv_hw
        self._load_w = load_w
        self._load_bn = load_bn
        self.prefix = f"features.{idx}"
        
    def forward(self, x, t, out_ch, stride):
        in_ch = x.shape[0]
        residual = x
        p = self.prefix.replace(".", "_")
        
        if t != 1:
            w = self._load_w(f"{p}_conv_0_0_weight")
            s = self._load_w(f"{p}_conv_0_0_weight_scale")[0]
            x = self.conv.conv1x1(x, w, s, stride=1)
            x = relu6(batch_norm(x, **self._load_bn(f"{p}_conv_0_1")))
            
            w = self._load_w(f"{p}_conv_1_0_weight")
            s = self._load_w(f"{p}_conv_1_0_weight_scale")[0]
            x = self.conv.depthwise_conv2d(x, w, s, stride=stride, padding=1)
            x = relu6(batch_norm(x, **self._load_bn(f"{p}_conv_1_1")))
            
            w = self._load_w(f"{p}_conv_2_weight")
            s = self._load_w(f"{p}_conv_2_weight_scale")[0]
            x = self.conv.conv1x1(x, w, s, stride=1)
            x = batch_norm(x, **self._load_bn(f"{p}_conv_3"))
        else:
            w = self._load_w(f"{p}_conv_0_0_weight")
            s = self._load_w(f"{p}_conv_0_0_weight_scale")[0]
            x = self.conv.depthwise_conv2d(x, w, s, stride=stride, padding=1)
            x = relu6(batch_norm(x, **self._load_bn(f"{p}_conv_0_1")))
            
            w = self._load_w(f"{p}_conv_1_weight")
            s = self._load_w(f"{p}_conv_1_weight_scale")[0]
            x = self.conv.conv1x1(x, w, s, stride=1)
            x = batch_norm(x, **self._load_bn(f"{p}_conv_2"))
            
        if stride == 1 and in_ch == out_ch:
            x = x + residual
        return x

# =====================================================================
# MODEL
# =====================================================================
class MobileNetV2_HW:
    def __init__(self, weight_dir, bitstream_path):
        self.weight_dir = weight_dir
        self.arch = [(1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,4,2),(6,96,3,1),(6,160,3,2),(6,320,1,1)]
        self.hw_mac = HardwareMAC3x3(bitstream_path)
        self.conv_hw = Conv3x3_HW(self.hw_mac)
        
    def _load_w(self, name):
        fp = os.path.join(self.weight_dir, name.replace(".","_") + ".npy")
        if not os.path.exists(fp):
            if "scale" in name: return np.array([1.0], dtype=np.float32)
            raise FileNotFoundError(f"Missing weight: {fp}")
        return np.load(fp, allow_pickle=False)
    
    def _load_bn(self, path):
        return {
            'gamma': self._load_w(f"{path}_weight"),
            'beta': self._load_w(f"{path}_bias"),
            'mean': self._load_w(f"{path}_running_mean"),
            'var': self._load_w(f"{path}_running_var")
        }
        
    def predict(self, img, verbose=True):
        x = img
        
        if verbose: print("  [Layer 0] Initial Conv 3x3...")
        w = self._load_w("features_0_0_weight")
        s = self._load_w("features_0_0_weight_scale")[0]
        x = self.conv_hw.conv2d(x, w, s, stride=2, padding=1)
        x = relu6(batch_norm(x, **self._load_bn("features_0_1")))
        if verbose: print(f"      shape: {x.shape}")
        
        idx = 1
        for (t,c,n,s) in self.arch:
            for b in range(n):
                stride = s if b==0 else 1
                if verbose:
                    print(f"  [Block {idx:2d}/17] t={t}, c={c}, s={stride} ... ", end="", flush=True)
                blk = InvertedResidualBlock(self.conv_hw, self._load_w, self._load_bn, idx)
                x = blk.forward(x, t, c, stride)
                if verbose: print(f"shape={x.shape}")
                idx += 1
                
        if verbose: print("  [Layer 18] Final Conv 1x1...")
        w = self._load_w("features_18_0_weight")
        s = self._load_w("features_18_0_weight_scale")[0]
        x = self.conv_hw.conv1x1(x, w, s, stride=1)
        x = relu6(batch_norm(x, **self._load_bn("features_18_1")))
        if verbose: print(f"      shape: {x.shape}")
        
        x = np.mean(x, axis=(1,2))
        w = self._load_w("classifier_1_weight")
        ws = self._load_w("classifier_1_weight_scale")[0]
        b = self._load_w("classifier_1_bias")
        logits = np.dot(w.astype(np.float32)/ws, x) + b.astype(np.float32)
        return logits
    
    def cleanup(self):
        self.hw_mac.cleanup()

# =====================================================================
# PREPROCESSING
# =====================================================================
def preprocess(img_path, size=64):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return np.transpose(arr, (2,0,1))

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("MOBILENETV2 FPGA INFERENCE - PYNQ-ZU")
    print("=" * 70)
    
    if not os.path.exists(BITSTREAM_PATH):
        print(f"[FATAL] Bitstream missing: {BITSTREAM_PATH}")
        return
    if not os.path.exists(WEIGHT_DIR):
        print(f"[FATAL] Weights missing: {WEIGHT_DIR}")
        return
    
    if os.path.exists(IMAGE_PATH):
        print(f"[Input] Loading: {IMAGE_PATH}")
        img = preprocess(IMAGE_PATH, size=64)
    else:
        print("[FATAL] No test image found.")
        return
    print(f"        shape: {img.shape}")
    
    model = MobileNetV2_HW(WEIGHT_DIR, BITSTREAM_PATH)
    
    try:
        print("\n[Inference] Starting...")
        t0 = time.perf_counter()
        logits = model.predict(img, verbose=True)
        t1 = time.perf_counter()
        
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETE")
        print(f"Total latency: {t1-t0:.1f}s")
        print("=" * 70)
        
        top5 = np.argsort(logits)[-5:][::-1]
        print("\nTop-5 Predictions:")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. Class {idx:4d}  (score: {logits[idx]:+.3f})")
 
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()


  (or)


"""
================================================================================
MOBILENETV2 FPGA INFERENCE - WITH FIXED WEIGHTS FOR PYNQ-ZU
================================================================================
"""

import numpy as np
from pynq import Overlay
from pynq import allocate
import os
from PIL import Image
import time

BITSTREAM_PATH = "./designn.bit"
WEIGHT_DIR = "/tmp/weights_fixed5"
IMAGE_PATH = "./picture.jpg"

# =====================================================================
# HARDWARE MAC (YOUR VERILOG MAC IN PL)
# =====================================================================
class HardwareMAC3x3:
    def __init__(self, bitstream_path):
        print(f"[HW] Loading bitstream: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)
        self.dma = self.overlay.axi_dma_0
        
        self.in_buffer = allocate(shape=(3,), dtype=np.uint64)
        self.out_buffer = allocate(shape=(1,), dtype=np.int32)
        print("[HW] Single-MAC DMA ready")
        
    def _pack(self, x_vals, y_vals):
        x = np.asarray(x_vals, dtype=np.int8).flatten()
        y = np.asarray(y_vals, dtype=np.int8).flatten()
        assert len(x) == 9 and len(y) == 9
        
        word0 = sum((int(x[i]) & 0xFF) << (8 * i) for i in range(8))
        word1 = (int(x[8]) & 0xFF)
        word1 |= sum((int(y[i]) & 0xFF) << (8 * (i + 1)) for i in range(7))
        word2 = (int(y[7]) & 0xFF) | ((int(y[8]) & 0xFF) << 8)
        
        return word0, word1, word2
        
    def run_mac(self, inputs, weights):
        w0, w1, w2 = self._pack(inputs, weights)
        self.in_buffer[0] = w0
        self.in_buffer[1] = w1
        self.in_buffer[2] = w2
        self.in_buffer.flush()
        
        self.dma.recvchannel.transfer(self.out_buffer)
        self.dma.sendchannel.transfer(self.in_buffer)
        
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        
        self.out_buffer.invalidate()
        return float(self.out_buffer[0])
    
    def cleanup(self):
        self.in_buffer.freebuffer()
        self.out_buffer.freebuffer()
        print("[HW] Buffers freed")

# =====================================================================
# FPGA CONV ENGINE
# =====================================================================
class Conv3x3_HW:
    def __init__(self, hw_mac):
        self.hw = hw_mac
        
    def _quantize(self, ifm):
        max_val = np.max(np.abs(ifm))
        scale = 127.0 / max_val if max_val > 1e-8 else 1.0
        q = np.clip(np.round(ifm * scale), -128, 127).astype(np.int8)
        return q, scale

    def conv2d(self, ifm, weights, weight_scale, stride=1, padding=1):
        out_ch, in_ch = weights.shape[0], weights.shape[1]
        padded = np.pad(ifm, ((0,0),(padding,padding),(padding,padding)), mode='constant') if padding else ifm
        out_h = (padded.shape[1] - 3) // stride + 1
        out_w = (padded.shape[2] - 3) // stride + 1
        ifm_q, input_scale = self._quantize(padded)
        out = np.zeros((out_ch, out_h, out_w), dtype=np.float32)
        deq = weight_scale * input_scale
        
        w_flat = {}
        for co in range(out_ch):
            for ci in range(in_ch):
                w_flat[(co, ci)] = weights[co, ci].flatten().tolist()
        
        for co in range(out_ch):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh*stride, ow*stride
                    acc = 0.0
                    for ci in range(in_ch):
                        acc += self.hw.run_mac(
                            ifm_q[ci, h0:h0+3, w0:w0+3].flatten().tolist(),
                            w_flat[(co, ci)]
                        )
                    out[co, oh, ow] = acc / deq
        return out

    def depthwise_conv2d(self, ifm, weights, weight_scale, stride=1, padding=1):
        ch = weights.shape[0]
        padded = np.pad(ifm, ((0,0),(padding,padding),(padding,padding)), mode='constant') if padding else ifm
        out_h = (padded.shape[1] - 3) // stride + 1
        out_w = (padded.shape[2] - 3) // stride + 1
        ifm_q, input_scale = self._quantize(padded)
        out = np.zeros((ch, out_h, out_w), dtype=np.float32)
        deq = weight_scale * input_scale
        
        w_flat = {}
        for c in range(ch):
            w_flat[c] = weights[c, 0].flatten().tolist()
        
        for c in range(ch):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh*stride, ow*stride
                    out[c,oh,ow] = self.hw.run_mac(
                        ifm_q[c, h0:h0+3, w0:w0+3].flatten().tolist(),
                        w_flat[c]
                    ) / deq
        return out

    def conv1x1(self, ifm, weights, weight_scale, stride=1):
        out_ch, in_ch = weights.shape[0], weights.shape[1]
        _, h, w = ifm.shape
        ifm_q, input_scale = self._quantize(ifm)
        out = np.zeros((out_ch, h, w), dtype=np.float32)
        deq = weight_scale * input_scale
        
        w_flat = {}
        for co in range(out_ch):
            for chunk in range(0, in_ch, 9):
                wv = weights[co, chunk:chunk+9, 0, 0]
                if len(wv) < 9:
                    wv = np.pad(wv, (0, 9-len(wv)), 'constant')
                w_flat[(co, chunk)] = wv.flatten().tolist()
        
        for co in range(out_ch):
            for i in range(h):
                for j in range(w):
                    acc = 0.0
                    for chunk in range(0, in_ch, 9):
                        xv = ifm_q[chunk:chunk+9, i, j]
                        if len(xv) < 9:
                            xv = np.pad(xv, (0, 9-len(xv)), 'constant')
                        acc += self.hw.run_mac(xv.flatten().tolist(), w_flat[(co, chunk)])
                    out[co, i, j] = acc / deq
        return out

# =====================================================================
# MOBILENETV2 BLOCKS
# =====================================================================
def relu6(x): return np.clip(x, 0.0, 6.0)

def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    g = gamma.astype(np.float32).reshape(-1,1,1)
    b = beta.astype(np.float32).reshape(-1,1,1)
    m = mean.astype(np.float32).reshape(-1,1,1)
    v = var.astype(np.float32).reshape(-1,1,1)
    return g * ((x - m) / np.sqrt(v + eps)) + b

class InvertedResidualBlock:
    def __init__(self, conv_hw, load_w, load_bn, idx):
        self.conv = conv_hw
        self._load_w = load_w
        self._load_bn = load_bn
        self.prefix = f"features.{idx}"
        
    def forward(self, x, t, out_ch, stride):
        in_ch = x.shape[0]
        residual = x
        p = self.prefix.replace(".", "_")
        
        if t != 1:
            w = self._load_w(f"{p}_conv_0_0_weight")
            s = self._load_w(f"{p}_conv_0_0_weight_scale")[0]
            x = self.conv.conv1x1(x, w, s, stride=1)
            x = relu6(batch_norm(x, **self._load_bn(f"{p}_conv_0_1")))
            
            w = self._load_w(f"{p}_conv_1_0_weight")
            s = self._load_w(f"{p}_conv_1_0_weight_scale")[0]
            x = self.conv.depthwise_conv2d(x, w, s, stride=stride, padding=1)
            x = relu6(batch_norm(x, **self._load_bn(f"{p}_conv_1_1")))
            
            w = self._load_w(f"{p}_conv_2_weight")
            s = self._load_w(f"{p}_conv_2_weight_scale")[0]
            x = self.conv.conv1x1(x, w, s, stride=1)
            x = batch_norm(x, **self._load_bn(f"{p}_conv_3"))
        else:
            w = self._load_w(f"{p}_conv_0_0_weight")
            s = self._load_w(f"{p}_conv_0_0_weight_scale")[0]
            x = self.conv.depthwise_conv2d(x, w, s, stride=stride, padding=1)
            x = relu6(batch_norm(x, **self._load_bn(f"{p}_conv_0_1")))
            
            w = self._load_w(f"{p}_conv_1_weight")
            s = self._load_w(f"{p}_conv_1_weight_scale")[0]
            x = self.conv.conv1x1(x, w, s, stride=1)
            x = batch_norm(x, **self._load_bn(f"{p}_conv_2"))
            
        if stride == 1 and in_ch == out_ch:
            x = x + residual
        return x

# =====================================================================
# MODEL
# =====================================================================
class MobileNetV2_HW:
    def __init__(self, weight_dir, bitstream_path):
        self.weight_dir = weight_dir
        self.arch = [(1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,4,2),(6,96,3,1),(6,160,3,2),(6,320,1,1)]
        self.hw_mac = HardwareMAC3x3(bitstream_path)
        self.conv_hw = Conv3x3_HW(self.hw_mac)
        
    def _load_w(self, name):
        fp = os.path.join(self.weight_dir, name.replace(".","_") + ".npy")
        if not os.path.exists(fp):
            if "scale" in name: return np.array([1.0], dtype=np.float32)
            raise FileNotFoundError(f"Missing weight: {fp}")
        return np.load(fp, allow_pickle=False)
    
    def _load_bn(self, path):
        return {
            'gamma': self._load_w(f"{path}_weight"),
            'beta': self._load_w(f"{path}_bias"),
            'mean': self._load_w(f"{path}_running_mean"),
            'var': self._load_w(f"{path}_running_var")
        }
        
    def predict(self, img, verbose=True):
        x = img
        
        if verbose: print("  [Layer 0] Initial Conv 3x3...")
        w = self._load_w("features_0_0_weight")
        s = self._load_w("features_0_0_weight_scale")[0]
        x = self.conv_hw.conv2d(x, w, s, stride=2, padding=1)
        x = relu6(batch_norm(x, **self._load_bn("features_0_1")))
        if verbose: print(f"      shape: {x.shape}")
        
        idx = 1
        for (t,c,n,s) in self.arch:
            for b in range(n):
                stride = s if b==0 else 1
                if verbose:
                    print(f"  [Block {idx:2d}/17] t={t}, c={c}, s={stride} ... ", end="", flush=True)
                blk = InvertedResidualBlock(self.conv_hw, self._load_w, self._load_bn, idx)
                x = blk.forward(x, t, c, stride)
                if verbose: print(f"shape={x.shape}")
                idx += 1
                
        if verbose: print("  [Layer 18] Final Conv 1x1...")
        w = self._load_w("features_18_0_weight")
        s = self._load_w("features_18_0_weight_scale")[0]
        x = self.conv_hw.conv1x1(x, w, s, stride=1)
        x = relu6(batch_norm(x, **self._load_bn("features_18_1")))
        if verbose: print(f"      shape: {x.shape}")
        
        x = np.mean(x, axis=(1,2))
        w = self._load_w("classifier_1_weight")
        ws = self._load_w("classifier_1_weight_scale")[0]
        b = self._load_w("classifier_1_bias")
        logits = np.dot(w.astype(np.float32)/ws, x) + b.astype(np.float32)
        return logits
    
    def cleanup(self):
        self.hw_mac.cleanup()

# =====================================================================
# PREPROCESSING
# =====================================================================
def preprocess(img_path, size=64):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return np.transpose(arr, (2,0,1))

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("MOBILENETV2 FPGA INFERENCE - PYNQ-ZU")
    print("=" * 70)
    
    if not os.path.exists(BITSTREAM_PATH):
        print(f"[FATAL] Bitstream missing: {BITSTREAM_PATH}")
        return
    if not os.path.exists(WEIGHT_DIR):
        print(f"[FATAL] Weights missing: {WEIGHT_DIR}")
        return
    
    if os.path.exists(IMAGE_PATH):
        print(f"[Input] Loading: {IMAGE_PATH}")
        img = preprocess(IMAGE_PATH, size=64)
    else:
        print("[FATAL] No test image found.")
        return
    print(f"        shape: {img.shape}")
    
    model = MobileNetV2_HW(WEIGHT_DIR, BITSTREAM_PATH)
    
    try:
        print("\n[Inference] Starting...")
        t0 = time.perf_counter()
        logits = model.predict(img, verbose=True)
        t1 = time.perf_counter()
        
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETE")
        print(f"Total latency: {t1-t0:.1f}s")
        print("=" * 70)
        
        top5 = np.argsort(logits)[-5:][::-1]
        print("\nTop-5 Predictions:")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. Class {idx:4d}  (score: {logits[idx]:+.3f})")
            
        IMAGENET_CLASSES = {
            207: "golden retriever",
            208: "Labrador retriever",
            250: "Siberian husky",
            281: "tabby cat",
            282: "tiger cat",
            954: "banana",
            963: "pizza",
            817: "sports car",
            403: "airliner",
            852: "tennis ball",
            700: "parachute",
            837: "sunglasses",
            805: "soccer ball",
            733: "pirate flag",
        }
        print("\nClass names:")
        for rank, idx in enumerate(top5, 1):
            name = IMAGENET_CLASSES.get(idx, "unknown")
            print(f"  {rank}. {name}")
            
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()


