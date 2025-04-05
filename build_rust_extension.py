"""
Build script for the Phynexus Rust extension.
"""

import os
import sys
import subprocess
import platform
from setuptools_rust import RustExtension, Binding

def build_rust_extension():
    """Build the Phynexus Rust extension."""
    print("Building Phynexus Rust extension...")
    
    try:
        subprocess.run(["rustc", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Rust is not installed. Please install Rust to build the extension.")
        return False
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    rust_path = os.path.join(current_dir, "src", "phynexus", "rust")
    
    if not os.path.exists(rust_path):
        print(f"Error: Rust extension not found at {rust_path}")
        return False
    
    try:
        target_dir = os.path.join(current_dir, "neurenix")
        os.makedirs(target_dir, exist_ok=True)
        
        cargo_cmd = ["cargo", "build", "--release"]
        
        features = ["python"]
        if platform.system() == "Linux":
            features.append("cpu")
        
        if features:
            cargo_cmd.extend(["--features", ",".join(features)])
        
        print(f"Running: {' '.join(cargo_cmd)} in {rust_path}")
        result = subprocess.run(cargo_cmd, cwd=rust_path, check=True)
        
        if result.returncode != 0:
            print(f"Error: Failed to build Rust extension. Return code: {result.returncode}")
            return False
        
        lib_name = "_phynexus"
        if platform.system() == "Windows":
            lib_ext = ".dll"
        elif platform.system() == "Darwin":
            lib_ext = ".dylib"
        else:
            lib_ext = ".so"
        
        import sysconfig
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        
        if ext_suffix:
            target_file = f"{lib_name}{ext_suffix}"
        else:
            target_file = f"{lib_name}{lib_ext}"
        
        lib_path = os.path.join(rust_path, "target", "release", f"lib{lib_name}{lib_ext}")
        target_path = os.path.join(target_dir, target_file)
        
        if os.path.exists(lib_path):
            import shutil
            shutil.copy2(lib_path, target_path)
            print(f"Copied {lib_path} to {target_path}")
        else:
            print(f"Error: Built library not found at {lib_path}")
            return False
        
        print("Phynexus Rust extension built successfully!")
        return True
    
    except Exception as e:
        print(f"Error building Rust extension: {e}")
        return False

if __name__ == "__main__":
    success = build_rust_extension()
    sys.exit(0 if success else 1)
