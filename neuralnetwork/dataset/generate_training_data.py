import random
import string
import pandas as pd
from typing import List
import requests
import re

# Constants
MIN_KEY_LENGTH = 2
MAX_KEY_LENGTH = 20
MAX_SHIFT = 100
CHUNK_SIZE = 7000
REPEATS_PER_KEY_LENGTH = 2
BOOK_LIMIT = 20
OUTPUT_FILE = "./neuralnetwork/dataset/vigenere_dataset.csv"

# Gutenberg scraping
def get_gutenberg_book_ids(limit=BOOK_LIMIT):
    url = "https://www.gutenberg.org/ebooks/search/?sort_order=downloads"
    response = requests.get(url)
    matches = re.findall(r'/ebooks/(\d+)', response.text)
    return list(dict.fromkeys(matches))[:limit]  # remove duplicates, limit

def download_plain_text(book_id: str) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Failed to fetch book {book_id}: {e}")
    return ""

def scrape_combined_chunks(chunk_size=CHUNK_SIZE, book_limit=BOOK_LIMIT):
    book_ids = get_gutenberg_book_ids(limit=book_limit)
    full_text = ""

    for book_id in book_ids:
        print(f"Fetching book {book_id}...")
        text = download_plain_text(book_id)
        full_text += "\n" + text.strip()

    # Clean up and chunk
    full_text = ''.join(filter(str.isprintable, full_text))
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i+chunk_size].strip()
        if len(chunk) >= chunk_size:
            chunks.append(chunk)
    print(f"Collected {len(chunks)} chunks of at least {chunk_size} characters.")
    return chunks

# Vigenère cipher
def vigenere_encrypt(plaintext: str, key: str) -> str:
    plaintext = ''.join(filter(str.isalpha, plaintext.upper()))
    key = key.upper()
    return ''.join(
        chr((ord(c) - 65 + ord(key[i % len(key)]) - 65) % 26 + 65)
        for i, c in enumerate(plaintext)
    )

# Coincidence count
def coincidence_count(ciphertext: str, max_shift: int) -> List[int]:
    return [
        sum(1 for i in range(len(ciphertext) - shift) if ciphertext[i] == ciphertext[i + shift])
        for shift in range(1, max_shift + 1)
    ]

# Random key
def generate_random_key(length: int) -> str:
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Dataset generation
def generate_dataset(texts: List[str]) -> pd.DataFrame:
    data = []

    for text in texts:
        cleaned = ''.join(filter(str.isalpha, text.upper()))
        if len(cleaned) < 1000:
            continue

        for key_len in range(MIN_KEY_LENGTH, MAX_KEY_LENGTH + 1):
            for _ in range(REPEATS_PER_KEY_LENGTH):
                sample = cleaned[:CHUNK_SIZE]
                key = generate_random_key(key_len)
                ciphertext = vigenere_encrypt(sample, key)
                vector = coincidence_count(ciphertext, MAX_SHIFT)
                data.append(vector + [key_len])

    columns = [f'shift_{i+1}' for i in range(MAX_SHIFT)] + ['key_length']
    return pd.DataFrame(data, columns=columns)

# Main
if __name__ == "__main__":
    print("Scraping texts...")
    chunks = scrape_combined_chunks(chunk_size=CHUNK_SIZE, book_limit=BOOK_LIMIT)

    print("Generating dataset...")
    dataset = generate_dataset(chunks)

    dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}. Total samples: {len(dataset)}")
