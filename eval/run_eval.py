from __future__ import annotations

import sys

from nsu_rag_llm.cli import app

if __name__ == "__main__":
    sys.exit(app(["eval", *sys.argv[1:]]))
