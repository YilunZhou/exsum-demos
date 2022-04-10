
try:
    import spacy
except:
    print('ERROR: The required "spacy" package is not found. ' + 
          'You can install it via "pip install spacy". After the installation, ' + 
          'you need to download the "en_core_web_trf" pre-trained model via ' + 
          '"python -m spacy download en_core_web_trf".')
    quit()
from spacy.tokens import Doc

spacy_api = None

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

def get_spacy_api():
    global spacy_api
    if spacy_api is None:
        spacy_api = SpacyAPI()
    return spacy_api

def parse_pos(sent):
    return get_spacy_api().parse_pos(sent)

def parse_ner(sent):
    return get_spacy_api().parse_ner(sent)

def parse_dep(sent):
    return get_spacy_api().parse_dep(sent)
