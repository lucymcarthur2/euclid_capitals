"""Entry point for the packaged app (PyInstaller)."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed so the matching workers start correctly in the .exe
    from initials_studio.__main__ import main
    main()
