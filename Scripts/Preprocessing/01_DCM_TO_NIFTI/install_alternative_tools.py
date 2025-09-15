#!/usr/bin/env python3
"""
Install alternative DICOM conversion tools for better success rates
"""

import subprocess
import sys

def install_alternative_tools():
    """Install alternative DICOM conversion tools"""
    
    print("=== Installing Alternative DICOM Conversion Tools ===")
    
    tools_to_install = [
        ("gdcmsan", "brew install gdcm"),
        ("dicom2", "pip install dicom2"),
        ("pydicom-tools", "pip install pydicom-tools")
    ]
    
    for tool_name, install_cmd in tools_to_install:
        print(f"\nChecking for {tool_name}...")
        
        # Check if already installed
        result = subprocess.run(["which", tool_name], capture_output=True)
        
        if result.returncode == 0:
            print(f"✓ {tool_name} is already installed")
        else:
            print(f"Installing {tool_name}...")
            print(f"Command: {install_cmd}")
            
            try:
                if install_cmd.startswith("brew"):
                    subprocess.run(install_cmd.split(), check=True)
                elif install_cmd.startswith("pip"):
                    subprocess.run(install_cmd.split(), check=True)
                
                print(f"✓ {tool_name} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {tool_name}: {e}")
                print(f"Please install manually: {install_cmd}")
    
    print("\n=== Installation Complete ===")
    print("Alternative tools available:")
    print("- gdcmsan: More permissive with orientation issues")
    print("- dicom2: Python-based DICOM converter")
    print("- pydicom-tools: Additional DICOM utilities")

if __name__ == "__main__":
    install_alternative_tools()
