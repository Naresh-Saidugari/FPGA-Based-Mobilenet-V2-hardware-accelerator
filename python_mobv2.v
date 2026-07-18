#!/usr/bin/env python3
"""
================================================================================
MOBILENETV2 FPGA INFERENCE - BATCHED DMA (CORRECTED)
================================================================================
"""

import numpy as np
from pynq import Overlay, allocate
from PIL import Image
import time
import os

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BITSTREAM_PATH = "/home/xilinx/jupyter_notebooks/naresh/design_2.bit"
WEIGHT_DIR     = "/tmp/weights_fixed5"
IMAGE_PATH     = "/home/xilinx/jupyter_notebooks/naresh/test.jpg"
MAX_BATCH      = 8192


# =====================================================================
# BATCHED HARDWARE MAC
# =====================================================================
class HardwareMAC3x3:
    def __init__(self, bitstream_path, max_batch=MAX_BATCH):
        print(f"[HW] Loading bitstream: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)
        self.dma = self.overlay.axi_dma_0
        self.max_batch = max_batch

        self.in_buf  = allocate(shape=(max_batch * 3,), dtype=np.uint64, cacheable=False)
        self.out_buf = allocate(shape=(max_batch,),     dtype=np.int32,  cacheable=False)
        print(f"[HW] Batched DMA ready (max_batch={max_batch})")

    def _pack_batch(self, x_batch, y_batch):
        x = x_batch.astype(np.uint8).astype(np.uint64)
        y = y_batch.astype(np.uint8).astype(np.uint64)

        w0 = (x[:,0]      ) | (x[:,1] <<  8) | (x[:,2] << 16) | (x[:,3] << 24) | \
             (x[:,4] << 32) | (x[:,5] << 40) | (x[:,6] << 48) | (x[:,7] << 56)
        w1 = (x[:,8]      ) | (y[:,0] <<  8) | (y[:,1] << 16) | (y[:,2] << 24) | \
             (y[:,3] << 32) | (y[:,4] << 40) | (y[:,5] << 48) | (y[:,6] << 56)
        w2 = (y[:,7]      ) | (y[:,8] <<  8)

        n = x_batch.shape[0]
        self.in_buf[0:n*3:3] = w0
        self.in_buf[1:n*3:3] = w1
        self.in_buf[2:n*3:3] = w2

    def run_batch(self, x_batch, y_batch):
        N = x_batch.shape[0]
        assert y_batch.shape[0] == N
        results = np.zeros(N, dtype=np.float32)

        for i in range(0, N, self.max_batch):
            end = min(i + self.max_batch, N)
            n = end - i

            self._pack_batch(x_batch[i:end], y_batch[i:end])

            self.dma.recvchannel.transfer(self.out_buf[:n])
            self.dma.sendchannel.transfer(self.in_buf[:n * 3])

            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()

            results[i:end] = self.out_buf[:n]

        return results

    def cleanup(self):
        self.in_buf.freebuffer()
        self.out_buf.freebuffer()
        print("[HW] Buffers freed")


# =====================================================================
# VECTORIZED CONV ENGINE
# =====================================================================
class Conv3x3_HW:
    def __init__(self, hw_mac):
        self.hw = hw_mac

    def _quantize(self, ifm):
        max_val = np.max(np.abs(ifm))
        scale = 127.0 / max_val if max_val > 1e-8 else 1.0
        q = np.clip(np.round(ifm * scale), -128, 127).astype(np.int8)
        return q, scale

    @staticmethod
    def _im2col(img, kernel_size, stride):
        from numpy.lib.stride_tricks import sliding_window_view
        H, W = img.shape
        windows = sliding_window_view(img, (kernel_size, kernel_size))
        windows = windows[::stride, ::stride]
        return windows.reshape(-1, kernel_size * kernel_size).T

    # -----------------------------------------------------------------
    # FIXED: reshape order matches loop nesting (ci outer, co middle)
    # -----------------------------------------------------------------
    def conv2d(self, ifm, weights, weight_scale, stride=1, padding=1):
        out_ch, in_ch = weights.shape[0], weights.shape[1]
        padded = np.pad(ifm, ((0,0),(padding,padding),(padding,padding)), mode='constant') if padding else ifm
        out_h = (padded.shape[1] - 3) // stride + 1
        out_w = (padded.shape[2] - 3) // stride + 1

        ifm_q, input_scale = self._quantize(padded)
        deq = weight_scale * input_scale
        w_flat = weights.reshape(out_ch, in_ch, 9).astype(np.int8)

        total_ops = out_ch * out_h * out_w * in_ch
        x_batch = np.zeros((total_ops, 9), dtype=np.int8)
        y_batch = np.zeros((total_ops, 9), dtype=np.int8)

        idx = 0
        for ci in range(in_ch):
            patches = self._im2col(ifm_q[ci], 3, stride)
            n = out_h * out_w
            for co in range(out_ch):
                x_batch[idx:idx+n] = patches.T
                y_batch[idx:idx+n] = w_flat[co, ci]
                idx += n

        results = self.hw.run_batch(x_batch, y_batch)
        # FIXED: (in_ch, out_ch, out_h, out_w) then sum over in_ch
        results = results.reshape(in_ch, out_ch, out_h, out_w)
        out = np.sum(results, axis=0) / deq
        return out

    def depthwise_conv2d(self, ifm, weights, weight_scale, stride=1, padding=1):
        ch = weights.shape[0]
        padded = np.pad(ifm, ((0,0),(padding,padding),(padding,padding)), mode='constant') if padding else ifm
        out_h = (padded.shape[1] - 3) // stride + 1
        out_w = (padded.shape[2] - 3) // stride + 1

        ifm_q, input_scale = self._quantize(padded)
        deq = weight_scale * input_scale
        w_flat = weights.reshape(ch, 9).astype(np.int8)

        total_ops = ch * out_h * out_w
        x_batch = np.zeros((total_ops, 9), dtype=np.int8)
        y_batch = np.zeros((total_ops, 9), dtype=np.int8)

        idx = 0
        for c in range(ch):
            patches = self._im2col(ifm_q[c], 3, stride)
            n = out_h * out_w
            x_batch[idx:idx+n] = patches.T
            y_batch[idx:idx+n] = w_flat[c]
            idx += n

        results = self.hw.run_batch(x_batch, y_batch)
        out = results.reshape(ch, out_h, out_w) / deq
        return out

    # -----------------------------------------------------------------
    # FIXED: reshape order matches loop nesting (co outer, chunk middle)
    # -----------------------------------------------------------------
    def conv1x1(self, ifm, weights, weight_scale, stride=1):
        out_ch, in_ch = weights.shape[0], weights.shape[1]
        _, h, w = ifm.shape
        ifm_q, input_scale = self._quantize(ifm)
        deq = weight_scale * input_scale

        chunks = (in_ch + 8) // 9
        pad_in_ch = chunks * 9

        w_padded = np.zeros((out_ch, pad_in_ch), dtype=np.float32)
        w_padded[:, :in_ch] = weights[:, :, 0, 0]
        w_flat = w_padded.reshape(out_ch, chunks, 9).astype(np.int8)

        a_padded = np.zeros((pad_in_ch, h, w), dtype=np.int8)
        a_padded[:in_ch] = ifm_q

        total_ops = out_ch * h * w * chunks
        x_batch = np.zeros((total_ops, 9), dtype=np.int8)
        y_batch = np.zeros((total_ops, 9), dtype=np.int8)

        idx = 0
        for co in range(out_ch):
            for ck in range(chunks):
                x_slice = a_padded[ck*9:(ck+1)*9].reshape(9, h*w).T
                n = h * w
                x_batch[idx:idx+n] = x_slice
                y_batch[idx:idx+n] = w_flat[co, ck]
                idx += n

        results = self.hw.run_batch(x_batch, y_batch)
        # FIXED: (out_ch, chunks, h, w) then sum over chunks
        results = results.reshape(out_ch, chunks, h, w)
        out = np.sum(results, axis=1) / deq
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
        self.arch = [(1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,4,2),
                     (6,96,3,1),(6,160,3,2),(6,320,1,1)]
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
            'beta' : self._load_w(f"{path}_bias"),
            'mean' : self._load_w(f"{path}_running_mean"),
            'var'  : self._load_w(f"{path}_running_var")
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
def preprocess(img_path, size=224):
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
    print("MOBILENETV2 FPGA INFERENCE - BATCHED DMA (CORRECTED)")
    print("=" * 70)

    if not os.path.exists(BITSTREAM_PATH):
        print(f"[FATAL] Bitstream missing: {BITSTREAM_PATH}")
        return
    if not os.path.exists(WEIGHT_DIR):
        print(f"[FATAL] Weights missing: {WEIGHT_DIR}")
        return

    if os.path.exists(IMAGE_PATH):
        print(f"[Input] Loading: {IMAGE_PATH}")
        img = preprocess(IMAGE_PATH, size=224)
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
