import re

import os
import requests
import shutil
from bs4 import BeautifulSoup
import progressbar


def download_file(url, path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def download_all(url, dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    r = requests.get(url)
    links = BeautifulSoup(r.text, 'html.parser').find_all('a')
    widgets = [
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    bar = progressbar.ProgressBar(widgets=widgets)
    for link in bar(links):
        path = os.path.join(dir, link.contents[0])
        if os.path.isfile(path):
            continue
        if re.match('.*\.tif{1,2}', link.contents[0]):
            download_file(link['href'], path)


def main():
    download_all("https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html", 'sat')
    download_all("https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html", 'map')


if __name__ == '__main__':
    main()
