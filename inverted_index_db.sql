-- DROP TABLE IF EXISTS main.IndexWord;
-- DROP TABLE IF EXISTS main.Posting;

CREATE TABLE IF NOT EXISTS IndexWord (
  word TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Posting (
  word TEXT NOT NULL,
  documentName TEXT NOT NULL,
  frequency INTEGER NOT NULL,
  indexes TEXT NOT NULL,
  PRIMARY KEY(word, documentName),
  FOREIGN KEY (word) REFERENCES IndexWord(word)
);