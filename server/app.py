"""Entry point for ``openenv serve``."""

from __future__ import annotations

import os
import uvicorn


def main():
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    main()
