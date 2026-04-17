import sys

from pothole_pipeline import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "online"])
    main()
