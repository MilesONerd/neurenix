"""
Test script to verify the Phynexus Rust extension.
"""

import sys

def test_phynexus_import():
    """Test importing the Phynexus Rust extension."""
    print("Testing Phynexus Rust extension import...")
    try:
        from neurenix import _phynexus
        print("✓ Phynexus Rust extension imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Error importing Phynexus Rust extension: {e}")
        return False

if __name__ == "__main__":
    print("Testing Phynexus Rust extension...")
    success = test_phynexus_import()
    sys.exit(0 if success else 1)
