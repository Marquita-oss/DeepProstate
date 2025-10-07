#!/usr/bin/env python3
"""
DeepProstate package entry point.
Allows running the application as: python -m deepprostate
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
