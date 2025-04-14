#!/bin/bash

# Script to install multilingual fonts for supporting Asian languages in matplotlib
# This handles most Asian scripts including Korean, Chinese, Japanese, Thai, etc.

set -e  # Exit on any error

echo "=== Installing fonts for multilingual text support in matplotlib ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux system
    echo "Detected Linux system"
    
    # Check if we're in a Docker container
    if [ -f /.dockerenv ]; then
        echo "Running in Docker container"
        
        # Update package lists
        apt-get update
        
        # Install common font packages
        apt-get install -y --no-install-recommends \
            fonts-noto-cjk \
            fonts-noto-cjk-extra \
            fonts-noto \
            fonts-nanum \
            fonts-wqy-zenhei \
            fonts-wqy-microhei \
            fonts-arphic-ukai \
            fonts-arphic-uming \
            fonts-thai-tlwg \
            fonts-dejavu-core \
            fontconfig
            
        # Clean up
        apt-get clean
        rm -rf /var/lib/apt/lists/*
    else
        # Regular Linux install
        if [ -f /etc/debian_version ]; then
            # Debian/Ubuntu
            sudo apt-get update
            sudo apt-get install -y \
                fonts-noto-cjk \
                fonts-noto-cjk-extra \
                fonts-noto \
                fonts-nanum \
                fonts-wqy-zenhei \
                fonts-wqy-microhei \
                fonts-arphic-ukai \
                fonts-arphic-uming \
                fonts-thai-tlwg \
                fonts-dejavu-core
        elif [ -f /etc/redhat-release ]; then
            # RHEL/CentOS/Fedora
            sudo dnf install -y \
                google-noto-cjk-fonts \
                google-noto-sans-fonts \
                google-noto-serif-fonts \
                wqy-zenhei-fonts \
                dejavu-sans-fonts
        elif [ -f /etc/arch-release ]; then
            # Arch Linux
            sudo pacman -S --noconfirm \
                noto-fonts \
                noto-fonts-cjk \
                noto-fonts-emoji \
                wqy-zenhei \
                ttf-dejavu
        fi
    fi
    
    # Rebuild font cache
    echo "Rebuilding font cache..."
    fc-cache -fv
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS system"
    
    if command -v brew &>/dev/null; then
        echo "Installing fonts using Homebrew..."
        brew tap homebrew/cask-fonts
        brew install --cask \
            font-noto-sans-cjk \
            font-noto-sans-cjk-sc \
            font-noto-sans-cjk-jp \
            font-noto-sans-cjk-kr \
            font-noto-sans-cjk-tc \
            font-noto-sans \
            font-nanum-gothic \
            font-source-han-sans
    else
        echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        echo "Or manually download and install the required fonts."
    fi
    
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows with MSYS2/MinGW/Git Bash
    echo "Detected Windows system"
    echo "On Windows, please manually download and install the following fonts:"
    echo "- Noto Sans CJK (for Chinese, Japanese, Korean): https://www.google.com/get/noto/help/cjk/"
    echo "- Noto Sans (for various scripts): https://www.google.com/get/noto/"
    echo "- Source Han Sans: https://github.com/adobe-fonts/source-han-sans/releases"
    
    # Alternative: Download some fonts automatically
    read -p "Would you like to download some fonts automatically? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p ~/fonts
        cd ~/fonts
        
        echo "Downloading Noto Sans CJK..."
        curl -L -o NotoSansCJK.zip https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK.ttc.zip
        
        echo "Downloading NanumGothic (Korean)..."
        curl -L -o NanumGothic.ttf https://github.com/naver/nanumfont/raw/master/ttf/NanumGothic.ttf
        
        echo "Please manually install the downloaded fonts from: ~/fonts"
    fi
else
    # Unknown OS
    echo "Unsupported operating system: $OSTYPE"
    echo "Please manually install multilingual fonts that support Asian languages."
fi

# Install fonts via pip for environments like conda
if command -v pip &>/dev/null; then
    echo "Installing font packages via pip (useful for conda environments)..."
    pip install matplotlib
    
    # Try to install fonts-conda if available
    pip install fonts-conda || echo "fonts-conda package not available, skipping"
fi

echo "=== Font installation completed ==="
echo "You may need to restart your application for the changes to take effect."
echo "If you still see missing glyphs, try manually installing additional fonts that support the specific language you need."

# Instructions for verifying font installation
echo "=== To verify font installation ==="
echo "Run the following Python code:"
echo "
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# List available fonts with 'Noto' in the name (these should support Asian scripts)
fonts = sorted([f.name for f in fm.fontManager.ttflist if 'noto' in f.name.lower()])
print('Available Noto fonts:', fonts)

# Test with some multilingual text
plt.figure(figsize=(10, 6))
plt.title('Multilingual Text Test')
plt.text(0.5, 0.7, '한국어 (Korean)', fontsize=20, ha='center')
plt.text(0.5, 0.5, '日本語 (Japanese)', fontsize=20, ha='center')
plt.text(0.5, 0.3, '中文 (Chinese)', fontsize=20, ha='center')
plt.axis('off')
plt.tight_layout()
plt.savefig('font_test.png')
plt.close()

print('Test image saved as font_test.png')
" 