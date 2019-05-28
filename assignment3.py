import re
import nltk
import logging
import sqlite3
import os
import time

from pathlib import Path
from typing import List, Iterable, Dict, Optional
from html2text import HTML2Text
from nltk.corpus import stopwords
from collections import defaultdict

# nltk.download('punkt')
# nltk.download('stopwords')

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

    def perform_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows


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

        # note: remove duplicates and tokens with length of 1
        return set(token for token in tokens if len(token) > 1)

    def remove_stopwords(self, tokens: Iterable[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]

    @property
    def stop_words(self):
        if self._stop_words is None:
            self._stop_words = self.load_stopwords()

        return self._stop_words

    @staticmethod
    def load_stopwords(stopwords_path='slovenian-stopwords.txt'):
        english = set(stopwords.words('english'))
        with open(stopwords_path, 'r') as fp:
            slovenian = set([line.strip() for line in fp.readlines()])
            return english.union(slovenian)

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
        return [str(match.start()) for match in re.finditer(word, content, re.IGNORECASE)]

    def create_index(self, preprocessor=Preprocessor(), db=DBHandler()):
        progress_counter = 0

        for file_name, document in self.corpus.items():
            progress_counter += 1
            logging.info(f'Progress: indexing {file_name} -> {progress_counter}/{len(self.corpus)}')

            for token in preprocessor(document):
                occurrences = self.find_occurrences(token, document)
                if len(occurrences) > 0:
                    db.insert(token, file_name, len(occurrences), ','.join(occurrences))


class QueryResults:
    def __init__(self, query: str, results: [str]):
        self.query = query
        self.results = results


class QueryResult:
    def __init__(self, count: int, query_results: [QueryResults]):
        self.count = count
        self.query_results = query_results


class SearchEngine:

    def __init__(self, db_handler: DBHandler):
        self.db = db_handler
        self.parser = HTML2Text()
        self.parser.ignore_links = True

    def perform_query(self, query: str):
        query_words = query.lower().split(" ")
        count = 0
        results = defaultdict(list)
        for query_word in query_words:
            rows = self.db.perform_query("SELECT * FROM Posting WHERE word='%s'" % query_word)
            for row in rows:
                document = row[1]
                for result in self._find_occurrences_in_file(row[1], self._to_postings_list(row[3]), query_word):
                    results[document].append(result)
                    count += 1
        query_results = [item for item in results.items()]
        query_results.sort(key=lambda x: len(x[1]), reverse=True)
        return count, query_results

    @staticmethod
    def _to_postings_list(postings: str):
        for i in postings.split(","):
            yield int(i)

    def _find_occurrences_in_file(self, file: str, postings_list: [int], token: str, surrounding_characters: int = 15):
        text = self._load_file(file)
        for index in postings_list:
            min_index = max(0, index - surrounding_characters)
            max_index = min(len(text), index + len(token) + surrounding_characters)
            yield "... " + text[min_index: max_index] + " ..."

    def _load_file(self, file):
        html = Path(file).read_text()
        return self.parser.handle(html)


def initiating_search(query: str):
    engine = SearchEngine(DBHandler())
    start = time.time()
    count, results = engine.perform_query(query)
    end = time.time()
    print('Searching time: ', end - start)
    print("Found %d results for \"%s\"" % (count, query))
    print()
    for result_query in results:
        print("Found %d results in \"%s\": " % (len(result_query[1]), result_query[0]))
        for idx, result in enumerate(result_query[1]):
          print(f'{idx}: {repr(result)}')
        print()


def initiating_indexing():
    start = time.time()
    google = BetterThanGoogle('data/')
    google.create_index()
    end = time.time()
    print('Indexing time: ', end - start)


if __name__ == '__main__':
    # initiating_indexing()
    for query in ["Sistem SPOT"]:
        initiating_search(query)
