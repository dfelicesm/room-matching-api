"""
pytest doesn’t know about our project root by default, my imports from src
were breaking.
This just adds the root to sys.path so tests run without needing PYTHONPATH=.
Not the prettiest, but it works :)
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
