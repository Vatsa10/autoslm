"""autoslm: closed-loop autonomous SLM adaptation.

Open re-implementation of Pioneer Agent (arXiv:2604.09791, Atreja et al., 2026).
"""
__version__ = "0.0.1"

# Auto-load .env from project root if present (provider keys, etc.).
try:
    from dotenv import load_dotenv as _load_dotenv
    from pathlib import Path as _Path
    for _p in (_Path.cwd() / ".env", _Path(__file__).resolve().parent.parent / ".env"):
        if _p.exists():
            _load_dotenv(_p, override=False)
            break
except ImportError:
    pass
