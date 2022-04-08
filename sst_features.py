
try:
    import spacy
except:
    print('ERROR: The required "spacy" package is not found. ' + 
          'You can install it via "pip install spacy". After the installation, ' + 
          'you need to download the "en_core_web_trf" pre-trained model via ' + 
          '"python -m spacy download en_core_web_trf".')
    quit()
from spacy.tokens import Doc
import datasets

spacy_api = None
word_sentiment_record = None

class SpacyAPI():
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except:
            print('ERROR: Featurization requires the "en_core_web_trf" pretrained ' + 
                  'model, which you can download via "python -m spacy download ' + 
                  'en_core_web_trf".')
            quit()
        self.cached_docs = dict()

    def get_doc(self, sent):
        assert isinstance(sent, tuple)
        if sent not in self.cached_docs:
            doc = Doc(self.nlp.vocab, sent)
            self.nlp(doc)
            self.cached_docs[sent] = doc
        else:
            doc = self.cached_docs[sent]
        return doc

    def parse_pos(self, sent):
        assert isinstance(sent, list), 'sentence must be a list of already tokenized words'
        doc = self.get_doc(tuple(sent))
        return [token.pos_ for token in doc]

    def parse_ner(self, sent):
        assert isinstance(sent, list), 'sentence must be a list of already tokenized words'
        doc = self.get_doc(tuple(sent))
        ners = []
        for token in doc:
            if token.ent_iob_ == 'O':
                ners.append('O')
            else:
                ners.append(f'{token.ent_iob_}-{token.ent_type_}')
        return ners

    def parse_dep(self, sent):
        assert isinstance(sent, list), 'sentence must be a list of already tokenized words'
        doc = self.get_doc(tuple(sent))
        return [f'{token.dep_} {token.head.i + 1}' for token in doc]

class WordSentimentRecord():
    def __init__(self):
        phrase_dict = datasets.load_dataset('sst', 'dictionary')['dictionary']
        self.sentiment_dict = {d['phrase']: d['label'] for d in phrase_dict if ' ' not in d['phrase']}

    def get_word_sentiment(self, word):
        if word not in self.sentiment_dict:
            raise Exception(f'Word {word} not found? ')
        else:
            return self.sentiment_dict[word] * 2 - 1

def get_spacy_api():
    global spacy_api
    if spacy_api is None:
        spacy_api = SpacyAPI()
    return spacy_api

def get_word_sentiment_record():
    global word_sentiment_record
    if word_sentiment_record is None:
        word_sentiment_record = WordSentimentRecord()
    return word_sentiment_record

def parse_pos(sent):
    return get_spacy_api().parse_pos(sent)

def parse_ner(sent):
    return get_spacy_api().parse_ner(sent)

def parse_dep(sent):
    return get_spacy_api().parse_dep(sent)

def get_word_sentiment(word):
    return get_word_sentiment_record().get_word_sentiment(word)
