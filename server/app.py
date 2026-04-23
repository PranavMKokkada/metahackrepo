"""Entry point for ``openenv serve``."""

from __future__ import annotations

import os
import uvicorn


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
