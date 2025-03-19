import nltk
from nltk.corpus import brown

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('brown')

def prepare_brown_corpus():
    """Prepare Brown corpus dataset"""
    print("Preparing Brown corpus...")
    sentences = brown.sents()
    text = ' '.join([' '.join(sentence) for sentence in sentences])
    with open('dataset.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved {len(sentences)} sentences to dataset.txt")

def main():
    download_nltk_data()
    prepare_brown_corpus()

if __name__ == "__main__":
    main() 