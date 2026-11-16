"""Start Initials Studio and open it in the browser:  python -m initials_studio [--port 8800] [--no-browser]"""
import argparse
import multiprocessing
import threading
import webbrowser


def main():
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser(prog="initials_studio")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    from .examples import install_examples
    install_examples()
    import uvicorn
    from .server import app
    url = f"http://127.0.0.1:{a.port}"
    print(f"\n  Initials Studio is running at {url}\n  (close this window to stop it)\n", flush=True)
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
