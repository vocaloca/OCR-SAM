#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Font Test Script - Verify multilingual font installation for matplotlib
This script tests if various Asian scripts can be rendered correctly
"""

import sys
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def main():
    print("\n=== Testing Font Installation for Asian Languages ===\n")
    
    # Get system info
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Force rebuild the font cache
    print("Rebuilding font cache...")
    fm._load_fontmanager(try_read_cache=False)
    
    # Directly add the WQY fonts to ensure they're available
    wqy_micro_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    wqy_zen_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    
    if os.path.exists(wqy_micro_path):
        fm.fontManager.addfont(wqy_micro_path)
        print(f"Added font: {wqy_micro_path}")
    
    if os.path.exists(wqy_zen_path):
        fm.fontManager.addfont(wqy_zen_path)
        print(f"Added font: {wqy_zen_path}")
    
    # 1. Get a list of all available fonts
    all_fonts = sorted([f.name for f in fm.fontManager.ttflist])
    print(f"Total fonts available: {len(all_fonts)}")
    
    # 2. Check for Asian language fonts
    asian_patterns = [
        'noto', 'cjk', 'korean', 'japanese', 'chinese', 'thai', 
        'nanum', 'wqy', 'han', 'ming', 'heiti', 'baekmuk',
        'arphic', 'lohit', 'meiryo', 'mingliu', 'gothic', 
        'wenquanyi', '文泉驛', '文泉驿'
    ]
    
    asian_fonts = []
    for font in fm.fontManager.ttflist:
        if any(pattern in font.name.lower() for pattern in asian_patterns):
            asian_fonts.append(font.name)
    
    print(f"\nFound {len(asian_fonts)} fonts that might support Asian languages:")
    for i, font in enumerate(sorted(set(asian_fonts[:20]))):  # Show first 20 only
        print(f"  {i+1}. {font}")
    
    if len(asian_fonts) > 20:
        print(f"  ... and {len(asian_fonts) - 20} more")
    
    if not asian_fonts:
        print("\n⚠️ WARNING: No Asian fonts found! Text rendering may not work properly.")
        print("Please install the necessary fonts using the install_asian_fonts.sh script.")
    
    # 3. Create test image with various scripts
    print("\nCreating test images with various Asian scripts...")
    
    scripts = {
        'Korean': '안녕하세요 (Hello)',
        'Japanese': 'こんにちは (Hello)',
        'Simplified Chinese': '你好 (Hello)',
        'Traditional Chinese': '你好 (Hello)',
        'Thai': 'สวัสดี (Hello)',
        'Hindi': 'नमस्ते (Hello)',
        'Arabic': 'مرحبا (Hello)',
        'Tamil': 'வணக்கம் (Hello)'
    }
    
    # Create a fontproperties object for WQY Micro Hei
    wqy_fp = None
    if os.path.exists(wqy_micro_path):
        wqy_fp = fm.FontProperties(fname=wqy_micro_path)
    
    # First test - all scripts with default font
    plt.figure(figsize=(10, 8))
    plt.title('Asian Language Font Test (Default Font)')
    
    y_pos = 0.9
    for script_name, text in scripts.items():
        plt.text(
            0.5, y_pos, f"{script_name}: {text}", 
            fontsize=14, ha='center'
        )
        y_pos -= 0.1
    
    plt.text(
        0.5, 0.1, f"Using font family: {plt.rcParams['font.family']}", 
        fontsize=10, ha='center', style='italic'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('font_test_default.png')
    plt.close()
    
    # Second test - with WQY fonts
    plt.figure(figsize=(10, 8))
    plt.title('Asian Language Font Test (Using WenQuanYi Font)')
    
    y_pos = 0.9
    for script_name, text in scripts.items():
        if wqy_fp:
            plt.text(
                0.5, y_pos, f"{script_name}: {text}", 
                fontsize=14, ha='center', fontproperties=wqy_fp
            )
        else:
            plt.text(
                0.5, y_pos, f"{script_name}: {text}", 
                fontsize=14, ha='center', family='WenQuanYi Micro Hei'
            )
        y_pos -= 0.1
    
    plt.text(
        0.5, 0.1, "Using WenQuanYi Micro Hei font", 
        fontsize=10, ha='center', style='italic'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('font_test_wqy.png')
    plt.close()
    
    print("\nTest images created:")
    print("  - font_test_default.png")
    print("  - font_test_wqy.png")
    
    print("\n=== Font Testing Complete ===")
    print("If you see missing characters (boxes or question marks) in the images,")
    print("you may need to install additional fonts for those specific languages.")


if __name__ == "__main__":
    main() 