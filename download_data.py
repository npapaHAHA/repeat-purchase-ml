from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen


DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"


def main():
    dataset = Path("data/retail.zip")
    if dataset.is_file() and dataset.stat().st_size:
        print(f"Using {dataset}")
        return

    dataset.parent.mkdir(parents=True, exist_ok=True)
    partial = dataset.with_suffix(".part")
    try:
        with urlopen(DATASET_URL, timeout=120) as response, partial.open("wb") as output:
            copyfileobj(response, output)
        partial.replace(dataset)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Downloaded {dataset}")


if __name__ == "__main__":
    main()
