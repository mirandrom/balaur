import datasets as ds
from collections import defaultdict

from typing import List


# unique identifiers for each dataset or concatenation of datasets
DATASET_LOADERS = {
    'wikitext-2': lambda: ds.load_dataset('wikitext', 'wikitext-2-raw-v1'),
    'wikitext-103': lambda: ds.load_dataset('wikitext', 'wikitext-103-raw-v1'),
    'wikipedia': lambda: ds.load_dataset('wikipedia', '20220301.en'),
    'bookcorpusopen': lambda: ds.load_dataset('bookcorpusopen', 'plain_text'),
}
CONCAT_DATASETS = {
    'wikibooks': ['wikipedia', 'bookcorpusopen'],
}
DATASET_NAMES = set(list(DATASET_LOADERS) + list(CONCAT_DATASETS))

# text columns for each dataset to filter unnecessary columns
TEXT_COL = 'text'
DATASET_TEXT_COLS = defaultdict(lambda: TEXT_COL)


def get_raw_dataset(dataset_name: str, split: float = None,
                    seed: int = 0xBE575EED) -> List[ds.DatasetDict]:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f'dataset_name `{dataset_name}` not in {DATASET_NAMES}')

    # load dataset(s)
    if dataset_name in CONCAT_DATASETS:
        datasets = [DATASET_LOADERS[d]() for d in CONCAT_DATASETS[dataset_name]]
    else:
        datasets = [DATASET_LOADERS[dataset_name]()]

    # filter out unnecessary columns
    for i in range(len(datasets)):
        for k, d in datasets[i].items():
            text_col = DATASET_TEXT_COLS[dataset_name]
            if text_col != TEXT_COL:
                d = d.rename_column(text_col, TEXT_COL)
            cols_to_remove = [c for c in d.column_names if c != TEXT_COL]
            datasets[i][k] = d.remove_columns(cols_to_remove)

    # split datasets
    if split is not None:
        for i in range(len(datasets)):
            datasets[i] = datasets[i]['train'].train_test_split(test_size=split, seed=seed)
            datasets[i]['validation'] = datasets[i].pop('test')

    # we don't concatenate datasets until after preprocessing
    # (to avoid e.g. one preprocess worker stuck on the books subset of wikibooks)

    # dataset = ds.DatasetDict()
    # dataset['train'] = ds.concatenate_datasets([d['train'] for d in datasets])
    # if split is not None:
    #     dataset['validation'] = ds.concatenate_datasets([d['validation'] for d in datasets])

    return datasets

