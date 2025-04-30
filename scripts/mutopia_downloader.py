import os
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Timothy Hyde 2025
# This script was used for scraping the all piano solo MIDI files from Mutopia.com



START_URL = "https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Piano&solo=on&max=9999"
DOWNLOAD_DIR = Path.home() / "Downloads" / "Mutopia_Piano_MIDI"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# I also want harpsichord and organ works, like from Bach.
ALLOWED_INSTRUMENTS = {
    "piano",
    "harpsichord, piano",
    "piano, organ",
    "harpsichord, organ, piano",
    "piano, harpsichord, organ"
}

def clean(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def get_all_entries_paginated():
    print("Scraping")
    options = Options()
    driver = webdriver.Chrome(options=options)

    all_entries = []
    startat = 0
    page_size = 10

    while True:
        url = f"https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Piano&solo=on&startat={startat}"
        print(f"\n Visiting page with start at={startat}")
        driver.get(url)
        time.sleep(3)

        tables = driver.find_elements(By.CSS_SELECTOR, "table.result-table")
        print(f"Found {len(tables)} entries.")

        if len(tables) == 0:
            print("No more entries found")
            break

        for t in tables:
            rows = t.find_elements(By.TAG_NAME, "tr")
            if len(rows) < 4:
                continue

            try:
                title = rows[0].find_elements(By.TAG_NAME, "td")[0].text.strip()
                composer_cell = rows[0].find_elements(By.TAG_NAME, "td")[1].text.strip()
                composer_name = composer_cell.split("(")[0].strip()
                composer_last = composer_name.split()[-1]
            except:
                continue

            instrument_text = rows[1].find_elements(By.TAG_NAME, "td")[0].text.strip().lower()
            instrument_text = instrument_text.replace("for ", "").strip()

            synonyms = {
                "orgue": "organ",
                "clavecin": "harpsichord"
            }
            for wrong, right in synonyms.items():
                instrument_text = instrument_text.replace(wrong, right)

            if instrument_text not in ALLOWED_INSTRUMENTS:
                continue

            midi_link = None
            for cell in rows[3].find_elements(By.TAG_NAME, "td"):
                try:
                    a = cell.find_element(By.TAG_NAME, "a")
                    href = a.get_attribute("href")
                    if href.endswith(".mid"):
                        midi_link = href
                        break
                except:
                    continue

            if midi_link:
                all_entries.append((title, composer_last, midi_link))

        startat += page_size
        time.sleep(2)

    driver.quit()
    print(f"\n Total entries: {len(all_entries)}")
    return all_entries


def download_midi(title, composer, link, index):
    filename = f"{index:04}_{clean(composer)}_{clean(title)}.mid"
    filepath = DOWNLOAD_DIR / filename

    print(f" Downloading: {filename}")
    r = requests.get(link)
    with open(filepath, "wb") as f:
        f.write(r.content)

def main():
    entries = get_all_entries_paginated()
    for idx, (title, composer, link) in enumerate(entries, start=1):
        try:
            download_midi(title, composer, link, idx)
            time.sleep(1)
        except Exception as e:
            print(f" Failed to download {title}: {e}")

    print("Complete")

if __name__ == "__main__":
    main()
