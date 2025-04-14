#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick script to check and update Matplotlib's font cache
"""

import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import subprocess


def main():
    # Force a rebuild of the font cache
    print("Rebuilding font cache...")
    fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    
    # Another approach is to try to use the system font config
    # fm._rebuild() - This doesn't exist in the current matplotlib version
    # Instead, we can use _load_fontmanager with try_read_cache=False
    fm._load_fontmanager(try_read_cache=False)
    
    # List all fonts
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"Total fonts detected: {len(all_fonts)}")
    
    # Look for WQY fonts specifically
    wqy_fonts = [
        f for f in all_fonts 
        if ('wqy' in f.lower() or 
            'wenquanyi' in f.lower() or 
            '文泉' in f)
    ]
    print("\nWQY Fonts detected:")
    for font in wqy_fonts:
        print(f"  - {font}")
    
    # Look for CJK fonts
    cjk_fonts = [
        f for f in all_fonts 
        if ('cjk' in f.lower() or 'noto' in f.lower())
    ]
    print("\nCJK/Noto Fonts detected:")
    for font in cjk_fonts[:10]:  # Show first 10
        print(f"  - {font}")
    
    if len(cjk_fonts) > 10:
        print(f"  ... and {len(cjk_fonts) - 10} more")
    
    # Try to get system fonts
    print("\nSystem fonts (from fc-list):")
    try:
        output = subprocess.check_output(
            "fc-list | grep -i 'wqy\\|wenquanyi' | head -5", 
            shell=True, 
            universal_newlines=True
        )
        print(output)
    except Exception as e:
        print(f"Error getting system fonts: {e}")
    
    # Create a simple test image with Chinese characters
    plt.figure(figsize=(10, 6))
    plt.title('WenQuanYi Font Test')
    
    # Try each detected WQY font
    if wqy_fonts:
        y_pos = 0.8
        for font in wqy_fonts:
            try:
                plt.text(
                    0.5, y_pos, f"你好 Hello using {font}", 
                    fontsize=14, ha='center', family=font
                )
                y_pos -= 0.1
            except Exception as e:
                print(f"Error with font {font}: {e}")
    else:
        # Direct approach with full path
        font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        if os.path.exists(font_path):
            try:
                prop = fm.FontProperties(fname=font_path)
                plt.text(
                    0.5, 0.5, "你好 Hello using WQY Micro Hei (direct path)", 
                    fontsize=14, ha='center', fontproperties=prop
                )
            except Exception as e:
                print(f"Error with direct font path: {e}")
        else:
            plt.text(
                0.5, 0.5, "No WQY fonts found", 
                fontsize=14, ha='center'
            )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('font_check.png')
    plt.close()
    
    print("\nFont check image saved as font_check.png")
    print("\nIf characters show as boxes or question marks,")
    print("the fonts are not properly configured.")


if __name__ == "__main__":
    main() 