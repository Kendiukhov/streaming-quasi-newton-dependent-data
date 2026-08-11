"""Download and unpack the two public hourly streams used in the paper.

Both are from the UCI Machine Learning Repository and are redistributed by UCI under
terms that permit research use; we download rather than vendor them.

  Metro Interstate Traffic Volume  (Hogue)     -> data/Metro_Interstate_Traffic_Volume.csv.gz
  Beijing Multi-Site Air Quality   (Zhang et al.) -> data/airq/PRSA_Data_*/PRSA_Data_<site>_*.csv
"""
import os
import sys
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
URLS = {
    "metro+interstate+traffic+volume.zip":
        "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
    "beijing+multi+site+air+quality+data.zip":
        "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
}


def main():
    os.makedirs(DATA, exist_ok=True)
    for fn, url in URLS.items():
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        with zipfile.ZipFile(path) as z:
            z.extractall(DATA)
    inner = os.path.join(DATA, "PRSA2017_Data_20130301-20170228.zip")
    if os.path.exists(inner):
        with zipfile.ZipFile(inner) as z:
            z.extractall(os.path.join(DATA, "airq"))
    need = [os.path.join(DATA, "Metro_Interstate_Traffic_Volume.csv.gz"),
            os.path.join(DATA, "airq", "PRSA_Data_20130301-20170228",
                         "PRSA_Data_Aotizhongxin_20130301-20170228.csv")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("MISSING after unpack:", missing, file=sys.stderr)
        sys.exit(1)
    print("data ready in", os.path.abspath(DATA))


if __name__ == "__main__":
    main()
