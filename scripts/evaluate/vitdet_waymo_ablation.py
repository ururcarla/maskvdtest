#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

"""Waymo评测入口（ablation 可控版本），复用 Argoverse Ablation 逻辑。"""
from vitdet_argoverse_ablation import main

if __name__ == "__main__":
    main()

