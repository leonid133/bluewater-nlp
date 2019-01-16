from pymystem3 import Mystem
import re
import logging

_WORD_CHAR_RE = re.compile(r'\w')


def is_word(token_str):
    return bool(_WORD_CHAR_RE.search(token_str))


class CAS(object):
    def __init__(self, input_text):
        self.input_text = input_text
        self.tokens = None
        self.sentences = None
        self.noun_phrases = None

    @property
    def lemma_poses(self):
        result = []
        for token_anno in self.tokens:
            if not token_anno.is_word:
                continue
            result_str = token_anno.lemma
            if token_anno.pos:
                result_str += '_' + token_anno.pos
            result.append(result_str)
        return result


class TokenAnnotation(object):
    def __init__(self, lemma, pos):
        self.lemma = lemma
        self.is_word = is_word(lemma)
        self.pos = pos
        self.grammemes = None

    def set_lemma(self, lemma):
        self.lemma = lemma
        self.is_word = is_word(lemma)

    def __str__(self):
        return "%s/%s/%s" % (self.lemma, self.pos, self.grammemes)


class TextAnalyzer:
    def process(self, cas):
        pass


class TextProcessingPipeline(TextAnalyzer):
    def __init__(self, *analyzers):
        self.analyzers = list(analyzers)
        if not self.analyzers:
            raise ValueError("No analyzer was given")

    def process(self, cas):
        for an in self.analyzers:
            an.process(cas)

    def process_txt(self, input_txt):
        cas = CAS(input_txt)
        self.process(cas)
        return cas


class MystemTextAnalyzer(TextAnalyzer):
    def __init__(self):
        self._mystem = Mystem()

    def process(self, cas):
        src_text = cas.input_text
        infos = self._mystem.analyze(src_text)
        cas.tokens = [tanno for tanno in (self._extract_token_anno(i) for i in infos) if tanno]

    _GRAMMEME_SEP_RE = re.compile('[,=|()]')
    _POS_RE = re.compile('^\w+')

    @classmethod
    def _extract_token_anno(cls, info):
        if 'analysis' in info:
            morph_arr = info['analysis']
            if morph_arr:
                morph_item = morph_arr[0]
                lemma = morph_item['lex']
                tag = morph_item['gr']
                pos = None
                if tag is not None:
                    pos = cls._extract_pos(tag)
                else:
                    # tag is None, so we should force the lower case
                    lemma = lemma.lower()
                # lemma = cls._join_lemma_pos(lemma, tag)
                token_anno = TokenAnnotation(lemma, pos)
                if tag:
                    token_anno.grammemes = set(cls._GRAMMEME_SEP_RE.split(tag))
                    if '' in token_anno.grammemes:
                        token_anno.grammemes.remove('')
                else:
                    token_anno.grammemes = set()
                return token_anno
        # in other cases: no analysis OR empty analysis results => fallback to original text
        lemma = info['text'].strip()
        if lemma:
            lemma = lemma.lower()
            token_anno = TokenAnnotation(lemma, None)
            token_anno.grammemes = set()
            return token_anno
        else:
            return None

    @classmethod
    def _extract_pos(cls, tag):
        pos_match = cls._POS_RE.search(tag)
        return pos_match.group(0) if pos_match else None


class PosMapper(TextAnalyzer):
    def __init__(self, mapping_path):
        log = logging.getLogger(__name__)
        self.mapping = {}
        with open(mapping_path) as inp:
            for line in inp:
                line = line.rstrip()
                src_pos, target_pos = line.split()
                self.mapping[src_pos] = target_pos
        log.info("Read PoS mapping from %s: %s", mapping_path, self.mapping)

    def process(self, cas):
        for t in cas.tokens:
            t.pos = self.mapping.get(t.pos)


def get_preprocessing_pipeline(mapping_path):
    return TextProcessingPipeline(
        MystemTextAnalyzer(),
        PosMapper(mapping_path)
    )
