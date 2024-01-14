import os
from pathlib import Path
from itertools import cycle
import pandas as pd
from pandas.errors import ParserError


def load_csv(filename: str):
    def _ensure_correct_separator(df: pd.DataFrame, filename: str, separator: str, encoding: str) -> pd.DataFrame:
        head_line = list(df.head(1))

        if len(head_line) == 1:
            correct_separator = ',' if separator == ';' else ';'

            df = pd.read_csv(filename, sep=correct_separator, encoding=encoding)
            head_line = list(df.head(1))
            if len(head_line) == 1:
                correct_separator = '\t'

                return pd.read_csv(filename, sep=correct_separator, encoding=encoding)

        return df

    if not os.path.isfile(filename):
        raise FileNotFoundError(f'File not found: {filename}')

    encodings = cycle(['utf8', 'cp1251'])
    separators = cycle([',', ';', '\t'])

    df = None
    encoding = next(encodings)
    separator = next(separators)

    for _ in range(10):
        try:
            df = pd.read_csv(filename, sep=separator, encoding=encoding)

            return _ensure_correct_separator(df, filename, separator, encoding)
        except Exception as e:
            if isinstance(e, UnicodeDecodeError):
                encoding = next(encodings)
            elif isinstance(e, ParserError):
                separator = next(separators)

    return df
