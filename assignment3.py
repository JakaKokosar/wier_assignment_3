import sqlite3
import glob
import re
import nltk

from typing import List, Iterable
from html2text import HTML2Text


nltk.download('punkt')
# conn = sqlite3.connect('inverted-index.db')


def load_corpus():
    _corpus = dict()

    for file in [file_path for file_path in glob.iglob('**/*.html', recursive=True)]:
        with open(file, 'r') as fp:
            _corpus[file] = fp.read()

    return _corpus


def load_stopwords():
    with open('slovenian-stopwords.txt', 'r') as fp:
        return set([line.strip() for line in fp.readlines()])


def remove_punctuation(raw_text: str):
    pattern = r'!|\"|#|$|%|&|\'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\|\]|^|_|`|{|\||}|~|\\'
    return re.sub(pattern, '', raw_text)


def to_lower_case(tokens: Iterable[str]) -> List[str]:
    return [token.lower() for token in tokens]


def remove_non_alphabetic(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token.isalpha()]


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token not in slovene_stopwords]


def text(html: str) -> str:
    parser = HTML2Text()
    parser.ignore_links = True
    return parser.handle(html)


def preprocess(raw_text: str) -> List[str]:
    cleaned_text = remove_punctuation(raw_text)
    tokens = nltk.tokenize.word_tokenize(cleaned_text)
    # apply all preprocess functions and remove duplicates using set()
    return list(set(remove_stopwords(to_lower_case(remove_non_alphabetic(tokens)))))


if __name__ == '__main__':
    slovene_stopwords = load_stopwords()
    corpus = load_corpus()

    print('Corpus loaded. Contains {} documents.'.format(len(corpus)))

    for content in corpus.values():
        print(preprocess(text(content)))







