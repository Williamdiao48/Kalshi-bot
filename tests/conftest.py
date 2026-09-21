"""Shared pytest setup.

Puts the repo root on sys.path so `import kalshi_bot...` works regardless of
the directory pytest is invoked from. The bot's modules import cleanly without
a .env (env-var accessors fall back to defaults), so no fixtures are needed
for the pure-function tests here.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
