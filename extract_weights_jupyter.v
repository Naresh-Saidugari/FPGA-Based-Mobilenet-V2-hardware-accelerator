"""
================================================================================
FIND AND EXTRACT weights_fixed5.zip FROM naresh FOLDER
================================================================================
"""

import os
import zipfile
import shutil

# Most likely path based on your description
ZIP_PATH = "/home/xilinx/jupyter_notebooks/naresh/weights_fixed5.zip"
ALT_PATH = "/home/xilinx/naresh/weights_fixed5.zip"

# Check which one exists
if os.path.exists(ZIP_PATH):
    print(f"✅ Found: {ZIP_PATH}")
    actual_path = ZIP_PATH
elif os.path.exists(ALT_PATH):
    print(f"✅ Found: {ALT_PATH}")
    actual_path = ALT_PATH
else:
    print("❌ Not found at expected paths. Searching...")
    import glob
    matches = glob.glob("/home/xilinx/**/weights_fixed5.zip", recursive=True)
    if matches:
        actual_path = matches[0]
        print(f"✅ Found: {actual_path}")
    else:
        print("❌ weights_fixed5.zip not found anywhere!")
        raise FileNotFoundError("Upload the zip to the naresh folder first")

print(f"\nFile size: {os.path.getsize(actual_path) / 1024:.1f} KB")

# Extract
EXTRACT_DIR = "/tmp/weights_fixed5"
os.makedirs(EXTRACT_DIR, exist_ok=True)

print(f"\nExtracting to {EXTRACT_DIR}...")
with zipfile.ZipFile(actual_path, 'r') as z:
    z.extractall(EXTRACT_DIR)
print("Done!")

# List extracted files
print("\n--- Extracted contents ---")
for root, dirs, files in os.walk(EXTRACT_DIR):
    level = root.replace(EXTRACT_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files[:5]:  # Show first 5 files
        print(f"{subindent}{f}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files)-5} more files") 
      

And to verify exported weights

      
      """
================================================================================
VERIFY EXPORTED WEIGHTS - NO PYTORCH NEEDED
================================================================================
"""

import numpy as np
import os

WEIGHT_DIR = "/tmp/weights_fixed5" # Change to your weights path

def check_file(path, expected_dtype=None, expected_shape=None, expected_range=None):
    """Check a single weight file."""
    if not os.path.exists(path):
        return False, "MISSING"
    
    data = np.load(path)
    issues = []
    
    if expected_dtype and data.dtype != expected_dtype:
        issues.append(f"dtype={data.dtype} (expected {expected_dtype})")
    
    if expected_shape and data.shape != expected_shape:
        issues.append(f"shape={data.shape} (expected {expected_shape})")
    
    if expected_range:
        mn, mx = data.min(), data.max()
        if mn < expected_range[0] or mx > expected_range[1]:
            issues.append(f"range=[{mn:.4f}, {mx:.4f}] (expected [{expected_range[0]}, {expected_range[1]}])")
    
    if issues:
        return False, ", ".join(issues)
    return True, f"OK  dtype={data.dtype}, shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]"

print("=" * 70)
print("WEIGHT VERIFICATION")
print("=" * 70)
print(f"Directory: {WEIGHT_DIR}\n")

# =====================================================================
# CHECK CONV WEIGHTS (should be int8 with scale file)
# =====================================================================
print("[1] Checking Conv Weights (should be int8 + have scale file)...")
conv_weights = [
    ("features_0_0_weight", (32, 3, 3, 3)),
    ("features_1_conv_0_0_weight", (32, 1, 3, 3)),
    ("features_1_conv_1_weight", (16, 32, 1, 1)),
    ("features_2_conv_0_0_weight", (96, 16, 1, 1)),
    ("features_2_conv_1_0_weight", (96, 1, 3, 3)),
    ("features_2_conv_2_weight", (24, 96, 1, 1)),
    ("features_18_0_weight", (1280, 320, 1, 1)),
    ("classifier_1_weight", (1000, 1280)),
]

all_ok = True
for name, expected_shape in conv_weights:
    w_path = os.path.join(WEIGHT_DIR, f"{name}.npy")
    s_path = os.path.join(WEIGHT_DIR, f"{name}_scale.npy")
    
    ok, msg = check_file(w_path, expected_dtype=np.int8, expected_shape=expected_shape, expected_range=(-128, 127))
    scale_ok = os.path.exists(s_path)
    
    status = "✓" if ok and scale_ok else "✗"
    print(f"  {status} {name}: {msg}")
    if not scale_ok:
        print(f"    ✗ Missing scale file: {name}_scale.npy")
    if not ok or not scale_ok:
        all_ok = False

# =====================================================================
# CHECK BN PARAMS (should be float32, NOT int8)
# =====================================================================
print("\n[2] Checking BN Params (should be float32, NOT int8)...")
bn_params = [
    "features_0_1_weight",      # gamma
    "features_0_1_bias",       # beta
    "features_0_1_running_mean",
    "features_0_1_running_var",
    "features_1_conv_0_1_weight",
    "features_1_conv_0_1_bias",
    "features_1_conv_0_1_running_mean",
    "features_1_conv_0_1_running_var",
    "features_1_conv_2_weight",
    "features_1_conv_2_bias",
    "features_1_conv_2_running_mean",
    "features_1_conv_2_running_var",
]

for name in bn_params:
    path = os.path.join(WEIGHT_DIR, f"{name}.npy")
    ok, msg = check_file(path, expected_dtype=np.float32)
    
    # Extra check: warn if values look like quantized int8 (all integers in -128 to 127)
    if ok:
        data = np.load(path)
        is_all_int = np.all(data == np.round(data))
        is_int8_range = data.min() >= -128 and data.max() <= 127
        if is_all_int and is_int8_range and data.max() > 10:
            msg += "  ⚠️ LOOKS LIKE INT8 (possible bug!)"
            all_ok = False
    
    status = "✓" if ok and "LOOKS LIKE INT8" not in msg else "✗"
    print(f"  {status} {name}: {msg}")

# =====================================================================
# CHECK SCALE VALUES
# =====================================================================
print("\n[3] Checking Scale Values (should be reasonable, e.g., 10-100)...")
scale_files = [
    "features_0_0_weight_scale",
    "features_1_conv_0_0_weight_scale",
    "features_1_conv_1_weight_scale",
    "features_18_0_weight_scale",
    "classifier_1_weight_scale",
]

for name in scale_files:
    path = os.path.join(WEIGHT_DIR, f"{name}.npy")
    if os.path.exists(path):
        scale = np.load(path)[0]
        status = "✓" if 1 < scale < 200 else "⚠️"
        print(f"  {status} {name}: {scale:.6f}")
    else:
        print(f"  ✗ {name}: MISSING")
        all_ok = False

# =====================================================================
# CHECK CLASSIFIER BIAS
# =====================================================================
print("\n[4] Checking Classifier Bias...")
bias_path = os.path.join(WEIGHT_DIR, "classifier_1_bias.npy")
ok, msg = check_file(bias_path, expected_dtype=np.float32)
print(f"  {'✓' if ok else '✗'} classifier_1_bias: {msg}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
if all_ok:
    print("RESULT: ALL CHECKS PASSED ✓")
    print("Weights look correct for FPGA inference.")
else:
    print("RESULT: SOME CHECKS FAILED ✗")
    print("Please re-export weights using the fixed script.")
print("=" * 70) to check correct weights extracted
