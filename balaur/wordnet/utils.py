from nltk.corpus.reader.wordnet import WordNetCorpusReader


def import_wordnet() -> WordNetCorpusReader:
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        import nltk
        nltk.download('wordnet')
        from nltk.corpus import wordnet as wn
    return wn

