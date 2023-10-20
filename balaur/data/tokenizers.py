import transformers as tr

TOKENIZER_LOADERS = {
    'gpt2': lambda: tr.AutoTokenizer.from_pretrained('gpt2'),
    'roberta-base': lambda: tr.AutoTokenizer.from_pretrained('roberta-base'),
    'bert-base-uncased': lambda: tr.AutoTokenizer.from_pretrained('bert-base-uncased'),
}


def get_tokenizer(tokenizer_name: str) -> tr.PreTrainedTokenizer:
    if tokenizer_name not in TOKENIZER_LOADERS:
        raise ValueError(f'Unknown tokenizer name `{tokenizer_name}` not in '
                         f'{list(TOKENIZER_LOADERS.keys())}')
    return TOKENIZER_LOADERS[tokenizer_name]()


SLOW_TOKENIZER_LOADERS = {
    'gpt2': lambda: tr.AutoTokenizer.from_pretrained('gpt2', use_fast=False),
    'roberta-base': lambda: tr.AutoTokenizer.from_pretrained('roberta-base', use_fast=False),
    'bert-base-uncased': lambda: tr.AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False),
}


def get_slow_tokenizer(tokenizer_name: str) -> tr.PreTrainedTokenizer:
    if tokenizer_name not in SLOW_TOKENIZER_LOADERS:
        raise ValueError(f'Unknown tokenizer name `{tokenizer_name}` not in '
                         f'{list(SLOW_TOKENIZER_LOADERS.keys())}')
    return SLOW_TOKENIZER_LOADERS[tokenizer_name]()
