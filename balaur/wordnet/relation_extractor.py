from collections import defaultdict

import transformers as tr
from nltk.corpus.reader.wordnet import Synset

from .utils import import_wordnet
from .stoplist import wn_stoplist
wn = import_wordnet()


from typing import Dict, List, Type
TokenT = str
VocabIdxT = int
VocabT = Dict[TokenT, VocabIdxT]
SynsetT = Type[Synset]
SynsetNameT = str  # string returned by `synset.name()` e.g. "dog.n.01"
TokenToSynsetNamesT = Dict[TokenT, List[SynsetNameT]]
SynsetNameToTokensT = Dict[SynsetNameT, List[TokenT]]
SynsetRelDictT = Dict[SynsetNameT, List[SynsetNameT]]  # dict for mapping synsets to their hypernyms/hyponyms/etc...
SynsetVocabT = Dict[SynsetNameT, VocabIdxT]
RelSynIdsT = List[int]  # list of related synset ids for a given token

"""
Our goal is to encode WordNet relations in a model's vocabulary.
However, WordNet relations are defined between synsets (synonym sets).
So first, we map the model's vocabulary (tokens) to synsets.
However, there are many noisy synsets in WordNet we do not want to capture.
So, we filter tokens and synsets that are likely to be noisy.
Note that one token may map to multiple synsets and one synset may have multiple tokens.

Next, we take the resulting synsets (tail) and get related synsets (head) in WordNet.
Now we have two options:
    - map the head embeddings back onto the vocabulary embeddings so we can 
        map relations between token-token pairs (limited by model vocabulary);
    - map relations between token-synset pairs (less limited by model vocabulary);
    
# TODO: finish writing this out, eventually
"""


# MBDO: refactor types, methods, and variables to be more consistent/self-documenting


class WordNetRelationExtractor:
    def __init__(self,
                 tokenizer_name: str,
                 min_tok_len: int = 3,
                 pos_keep: str = 'n',  # 'nvars'
                 lexname_filter: str = None,
                 hypernymy: bool = True,
                 hyponymy: bool = True,
                 synonymy: bool = True,
                 antonymy: bool = True,
                 rel_depth: int = 1,
                 ):
        self.tokenizer_name = tokenizer_name
        self.min_tok_len = min_tok_len
        self.pos_keep = pos_keep
        self.lexname_filter = lexname_filter or ['noun.quantity', 'noun.motive',
                                                 'noun.shape', 'noun.relation',
                                                 'noun.process']
        self.hypernymy = hypernymy
        self.hyponymy = hyponymy
        self.synonymy = synonymy
        self.antonymy = antonymy
        self.rel_depth = rel_depth

        self.vocab = self.load_vocab()
        self.token_to_synsets = self.map_token_to_synsets(self.vocab)
        self.synset_to_tokens = self.map_synset_to_tokens(self.token_to_synsets)
        self.related_synsets = self.map_related_synsets(self.synset_to_tokens)
        self.filter_related_synsets()
        self.relations = sorted(self.rel2tok2syn.keys())


    def load_vocab(self) -> VocabT:
        tokenizer = tr.AutoTokenizer.from_pretrained(self.tokenizer_name)
        vocab = tokenizer.get_vocab()
        if any([t in self.tokenizer_name for t in ['roberta', 'gpt']]):
            vocab = self.preproc_gpt_style_vocab(vocab)
        vocab = self.filter_vocab(vocab)
        return vocab

    def preproc_gpt_style_vocab(self, vocab: VocabT) -> VocabT:
        """We want to filter wordpiece tokens and keep only whole-word tokens.
        In GPT-style vocabs, tokens preceded by whitespace in the corpus are
        prefixed with `Ġ` while tokens preceded by other tokens (ie wordpieces)
        are not prefixed.

        Thus, we remove the `Ġ` prefix to obtain whole-word tokens that can
        be searched in wordnet, and otherwise add a `##` prefix to identify
        wordpieces consistent with the BERT vocab.
        """
        new_vocab = {}
        for tok, idx in vocab.items():
            if tok.startswith('Ġ'):
                tok = tok[1:]
            else:
                tok = f'##{tok}'
            new_vocab[tok] = idx
        return new_vocab

    def filter_vocab(self, vocab: VocabT) -> VocabT:
        new_vocab = {}
        for tok, idx in vocab.items():
            if tok.startswith('##'):  # remove wordpieces
                continue
            elif tok.lower() in wn_stoplist:
                continue
            elif not tok.isalpha():
                continue
            elif len(tok) < self.min_tok_len:
                continue
            new_vocab[tok] = idx
        return new_vocab

    def map_token_to_synsets(self, vocab: VocabT) -> TokenToSynsetNamesT:
        token_to_synsets = {}
        for tok, idx in vocab.items():
            synsets = [s.name() for s in wn.synsets(tok) if not self.filter_synset(s)]
            if len(synsets) == 0:
                continue
            token_to_synsets[tok] = synsets
        return token_to_synsets

    def filter_synset(self, synset: SynsetT) -> bool:
        if synset.pos() not in self.pos_keep:
            return True
        if synset.name().split(".")[0] in wn_stoplist:
            return True
        if synset.lexname() in self.lexname_filter:
            return True
        if len(synset.hypernyms()) == 0:
            return True
        return False

    def map_synset_to_tokens(self, token_to_synsets: TokenToSynsetNamesT) -> SynsetNameToTokensT:
        synset_to_tokens = defaultdict(list)
        for tok, synsets in token_to_synsets.items():
            for synset in synsets:
                synset_to_tokens[synset].append(tok)
        return synset_to_tokens

    def map_related_synsets(self, synset_to_tokens: SynsetNameToTokensT):
        related_synsets = {}
        if self.hypernymy:
            related_synsets['hypernymy'] = self.map_synset_hypernymy(synset_to_tokens)
        if self.hyponymy:
            related_synsets['hyponymy'] = self.map_synset_hyponymy(synset_to_tokens)
        if self.synonymy:
            related_synsets['synonymy'] = self.map_synset_synonymy(synset_to_tokens)
        if self.antonymy:
            related_synsets['antonymy'] = self.map_synset_antonymy(synset_to_tokens)
        return related_synsets

    @staticmethod
    def get_synset_hypernyms(sn: SynsetNameT, depth: int = 1, identity: bool = False) -> List[SynsetNameT]:
        hypernyms = set()
        curr_depth_hypernyms = [sn]
        while depth >= 1:
            curr_depth_hypernyms = [h2.name()
                                    for h1 in curr_depth_hypernyms
                                    for h2 in wn.synset(h1).hypernyms()]
            # handle reaching root of tree, or cycles
            curr_depth_hypernyms = [h for h in curr_depth_hypernyms if h not in hypernyms]
            if len(curr_depth_hypernyms) == 0:
                break
            hypernyms.update(curr_depth_hypernyms)
            depth -= 1

        if identity:
            hypernyms.add(sn)

        return list(hypernyms)

    @staticmethod
    def get_synset_hyponyms(sn: SynsetNameT, depth: int = 1, identity: bool = False) -> List[SynsetNameT]:
        hyponyms = set()
        curr_depth_hyponyms = [sn]
        while depth >= 1:
            curr_depth_hyponyms = [h2.name()
                                   for h1 in curr_depth_hyponyms
                                   for h2 in wn.synset(h1).hyponyms()]
            # handle reaching root of tree, or cycles
            curr_depth_hyponyms = [h for h in curr_depth_hyponyms if h not in hyponyms]
            if len(curr_depth_hyponyms) == 0:
                break
            hyponyms.update(curr_depth_hyponyms)
            depth -= 1

        if identity:
            hyponyms.add(sn)

        return list(hyponyms)

    def map_synset_hypernymy(self, synset_to_tokens: SynsetNameToTokensT):
        synset_hypernymy = {}
        for synset, tokens in synset_to_tokens.items():
            hypernyms = self.get_synset_hypernyms(synset, identity=False, depth=self.rel_depth)
            if len(hypernyms) == 0:
                continue
            synset_hypernymy[synset] = hypernyms
        return synset_hypernymy

    def map_synset_hyponymy(self, synset_to_tokens: SynsetNameToTokensT):
        synset_hyponymy = {}
        for synset, tokens in synset_to_tokens.items():
            hyponyms = self.get_synset_hyponyms(synset, identity=False, depth=1)
            if len(hyponyms) == 0:
                continue
            synset_hyponymy[synset] = hyponyms
        return synset_hyponymy

    def map_synset_synonymy(self, synset_to_tokens: SynsetNameToTokensT):
        synset_synonymy = {}
        for synset, tokens in synset_to_tokens.items():
            synset_synonymy[synset] = [synset]
        return synset_synonymy

    def map_synset_antonymy(self, synset_to_tokens: SynsetNameToTokensT):
        synset_antonymy = {}
        for synset, tokens in synset_to_tokens.items():
            antonyms = [a.synset().name() for l in wn.synset(synset).lemmas() for a in l.antonyms()]
            if len(antonyms) == 0:
                continue
            synset_antonymy[synset] = antonyms
        return synset_antonymy

    def filter_related_synsets(self):
        """
        Implement the following criteria to filter related synsets:
        - keep synset if it's an antonym and synonym (i.e. already in vocab);
        - keep synset if it's a synonym of at least `x` tokens
        - keep synset if it's a hypernym/hyponym of at least `y` head synsets

        Lastly, build a vocabulary of remaining synsets (i.e. map to index)
        """
        # first figure out which synsets to keep
        keep_synsets = set()
        if self.antonymy:
            for head_synset, rel_synsets in self.related_synsets['antonymy'].items():
                for s in rel_synsets:
                    if s in self.synset_to_tokens:
                        keep_synsets.add(s)
        if self.synonymy:
            for head_synset, rel_synsets in self.related_synsets['synonymy'].items():
                for s in rel_synsets:
                    num_tokens = len(self.synset_to_tokens.get(s, []))
                    if  num_tokens >= 2:
                        keep_synsets.add(s)

        if self.hypernymy or self.hyponymy:
            # first map each hypernym/hyponym synset (head) to it's tail synsets
            rel_hypernyms = self.related_synsets.get('hypernymy', {})
            hypernyms = {}
            for hypo, hypers in rel_hypernyms.items():
                for hyper in hypers:
                    hypernyms[hyper] = hypernyms.get(hyper, set()) | {hypo}

            rel_hyponyms = self.related_synsets.get('hyponymy', {})
            hyponyms = {}
            for hyper, hypos in rel_hyponyms.items():
                for hypo in hypos:
                    hyponyms[hypo] = hyponyms.get(hypo, set()) | {hyper}

            # now only keep synsets for which the tail synsets exceed a threshold
            for hypo, hypers in rel_hypernyms.items():
                for hyper in hypers:
                    num_synsets = len(hypernyms.get(hyper, [])) + len(hyponyms.get(hyper, []))
                    if num_synsets >= 2:
                        keep_synsets.add(hyper)

            for hyper, hypos in rel_hyponyms.items():
                for hypo in hypos:
                    num_synsets = len(hyponyms.get(hypo, [])) + len(hypernyms.get(hypo, []))
                    if num_synsets >= 2:
                        keep_synsets.add(hypo)

        # next filter these out
        for rel, related_synsets in self.related_synsets.items():
            new_related_synsets = {}
            for synset, rel_synsets in related_synsets.items():
                rel_synsets = [s for s in rel_synsets if s in keep_synsets]
                if rel_synsets:
                    new_related_synsets[synset] = rel_synsets
            self.related_synsets[rel] = new_related_synsets

        # lastly create the vocab and use this to create a map of related token-synset index pairs
        self.synset_vocab = {s: i for i, s in enumerate(sorted(keep_synsets))}
        self.rel2tok2syn = {rel: defaultdict(list) for rel in self.related_synsets.keys()}
        for token, synsets in self.token_to_synsets.items():
            tok_idx = self.vocab[token]
            for rel, related_synsets in self.related_synsets.items():
                synset_idxs = set()
                for synset in synsets:
                    for related_synset in related_synsets.get(synset, []):
                        synset_idxs.add(self.synset_vocab[related_synset])
                self.rel2tok2syn[rel][tok_idx] = list(synset_idxs)

        # also create a map that indexes token then relation
        self.tok2rel2syn = defaultdict(dict)
        for rel, tok2syn in self.rel2tok2syn.items():
            for tok, syns in tok2syn.items():
                self.tok2rel2syn[tok][rel] = syns

    def get_tok_related(self, tok_idx: VocabIdxT) -> Dict[str, RelSynIdsT]:
        return self.tok2rel2syn[tok_idx]







