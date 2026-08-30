"""
conftest.py
------------
Makes sure the project root (this file's directory) is on sys.path, so
files under tests/ can `import gesture`, `import config`, etc. regardless
of the working directory pytest is run from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
