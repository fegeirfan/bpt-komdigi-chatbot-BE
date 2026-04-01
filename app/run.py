import os

import uvicorn


def main() -> None:
    port = int((os.getenv("PORT") or "8000").strip())
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

