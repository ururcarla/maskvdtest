#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

"""Waymo评测入口（mask + bytetrack + dynmask_safe），逻辑同 Argoverse 版本。"""
from vitdet_argoverse_mask_bytetrack_dynmask_safe import main

if __name__ == "__main__":
    main()
