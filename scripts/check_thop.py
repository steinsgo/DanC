# Check ultralytics-thop dependency necessity
# ultralytics-thop is used for model profiling (FLOPs/params calculation)
# It's NOT required for training, only for model.info() calls

# Solution: Install ultralytics without ultralytics-thop dependency
# Then monkey-patch the profiling function to skip thop usage

import sys
import importlib.util

def check_thop_usage():
    """Check if thop is actually imported during YOLO training"""
    try:
        from ultralytics import YOLO
        print("✓ ultralytics imported successfully")

        # Check if thop is imported
        if 'thop' in sys.modules:
            print("✗ thop is imported")
        else:
            print("✓ thop is NOT imported (good!)")

        # Check if ultralytics.utils.torch_utils uses thop
        spec = importlib.util.find_spec('ultralytics.utils.torch_utils')
        if spec:
            print(f"✓ Found torch_utils at: {spec.origin}")
            with open(spec.origin, 'r') as f:
                content = f.read()
                if 'thop' in content or 'profile' in content:
                    print("⚠ torch_utils mentions thop/profile")
                else:
                    print("✓ torch_utils doesn't use thop")
    except ImportError as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    check_thop_usage()
