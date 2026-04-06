"""Launch the FastAPI server.

Run this from PyCharm (right-click -> Run). Then open:
    http://127.0.0.1:8000/docs
for the Swagger UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from src.config import API_HOST, API_PORT


def main() -> None:
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
