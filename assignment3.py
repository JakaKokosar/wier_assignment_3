import re
import nltk
import logging
import sqlite3

from pathlib import Path
from typing import List, Iterable, Dict, Optional
from html2text import HTML2Text

# nltk.download('punkt')

logging.getLogger().setLevel(logging.INFO)


class DBHandler:
    db_file_name = 'inverted-index.db'

    def __init__(self):
        self.connection = sqlite3.connect(self.db_file_name)
        self.create_table()

    def __del__(self):
        """ this is called when an object is garbage collected """
        self.connection.close()

    def create_table(self):
        cursor = self.connection.cursor()
        with open('inverted_index_db.sql', 'r') as fp:
            cursor.executescript(fp.read())

        cursor.close()

    def insert(self, word, document, freq, occurrences):
        cursor = self.connection.cursor()

        cursor.execute('INSERT OR IGNORE INTO IndexWord(word) VALUES(?)', (word,))
        cursor.execute('INSERT INTO Posting(word, documentName, frequency, indexes) VALUES(?, ?, ?, ?)',
                       (word, document, freq, occurrences))
        cursor.close()

        self.connection.commit()


class Preprocessor:

    def __init__(self):
        self._stop_words = None

    def __call__(self, text):
        """
        When calling preprocessor on a given text we get a list of tokens
        which are ready to be indexed.
        """
        text = self.remove_punctuation(text)

        # note: tokenize text after removing punctuation
        tokens = nltk.tokenize.word_tokenize(text)

        # todo: do we keep numbers in text?
        tokens = self.remove_non_alphabetic(tokens)

        # note: apply lower case transformation before removing stopwords
        tokens = self.to_lower_case(tokens)
        tokens = self.remove_stopwords(tokens)

        # note: remove duplicates
        return list(set(tokens))

    def remove_stopwords(self, tokens: Iterable[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]

    @property
    def stop_words(self):
        if self._stop_words is None:
            self._stop_words = self.load_stopwords()

        return self._stop_words

    @staticmethod
    def load_stopwords(stopwords_path='slovenian-stopwords.txt'):
        with open(stopwords_path, 'r') as fp:
            return set([line.strip() for line in fp.readlines()])

    @staticmethod
    def remove_punctuation(raw_text: str):
        pattern = r'!|\"|#|$|%|&|\'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\|\]|^|_|`|{|\||}|~|\\'
        return re.sub(pattern, '', raw_text)

    @staticmethod
    def to_lower_case(tokens: Iterable[str]) -> List[str]:
        return [token.lower() for token in tokens]

    @staticmethod
    def remove_non_alphabetic(tokens: Iterable[str]) -> List[str]:
        return [token for token in tokens if token.isalpha()]


class BetterThanGoogle:

    def __init__(self, dir_path: str):
        """
        :param dir_path: path to the directory of content we want to index.
        """

        self._corpus: Optional[Dict[str, str]] = None
        self.corpus_dir_path: str = dir_path

    @property
    def corpus(self):
        if self._corpus is None:
            self._corpus = self._load_corpus()

        return self._corpus

    def _load_corpus(self) -> Dict[str, str]:
        logging.info('Loading corpus intro memory ...')

        def text(html: str) -> str:
            """ Extract all text found between HTML tags """
            parser = HTML2Text()
            parser.ignore_links = True
            return parser.handle(html)

        return {str(path): text(path.read_text()) for path in Path(self.corpus_dir_path).glob('**/*.html')}

    @staticmethod
    def find_occurrences(word: str, content: str) -> List[str]:
        return [str(match.start()) for match in re.finditer(word, content)]

    def create_index(self, preprocessor=Preprocessor(), db=DBHandler()):
        progress_counter = 0

        for file_name, document in self.corpus.items():
            progress_counter += 1
            logging.info(f'Progress: indexing {file_name} -> {progress_counter}/{len(self.corpus)}')

            for token in preprocessor(document):
                occurrences = self.find_occurrences(token, document)
                if len(occurrences) > 0:
                    db.insert(token, file_name, len(occurrences), ','.join(occurrences))


if __name__ == '__main__':
    google = BetterThanGoogle('data/')
    google.create_index()
