import os

import uvicorn


def main() -> None:
    port = int((os.getenv("PORT") or "8000").strip())
    log_level = (os.getenv("LOG_LEVEL") or "info").strip().lower() or "info"
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        access_log=True,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
