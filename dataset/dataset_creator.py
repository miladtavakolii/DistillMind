import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


class DatasetMerger:
    '''
    Merge CSV, TSV, JSON, and JSONL datasets containing text and labels.

    The class loads datasets from a directory, merges them,
    applies preprocessing, removes duplicates, and saves output
    as JSONL.
    '''

    def __init__(
        self,
        input_dir: str,
        output_path: str,
        text_cols: List[str],
    ) -> None:
        '''
        Initialize merger configuration.

        Args:
            input_dir: Directory containing dataset files.
            output_path: Path to output JSONL file.
            text_cols: Columns name containing text.
        '''
        self.input_dir: Path = Path(input_dir)
        self.output_path: str = output_path
        self.text_cols: List[str] = text_cols

    def preprocess_text(self, text: object) -> str:
        '''
        Clean and normalize text.

        Args:
            text: Raw text value.

        Returns:
            Preprocessed text string.
        '''
        if not isinstance(text, str):
            return ''
        html_pattern = re.compile(r'<.*?>')
        emoji_pattern = re.compile(r'[^\w\sآ-ی]')
        space_pattern = re.compile(r'\s+')

        # Remove HTML tags
        text = html_pattern.sub(' ', text)

        # Remove emojis & non-Persian characters
        text = emoji_pattern.sub(' ', text)

        # Remove extra spaces
        text = space_pattern.sub(' ', text).strip()

        return text

    def load_file(self, path: Path) -> Optional[pd.DataFrame]:
        '''
        Load dataset file based on extension.

        Args:
            path: Path to dataset file.

        Returns:
            DataFrame with selected columns or None if invalid.
        '''
        suffix: str = path.suffix.lower()

        try:
            if suffix == '.csv':
                df = pd.read_csv(path)

            elif suffix == '.tsv':
                df = pd.read_csv(path, sep='\t')

            elif suffix == '.jsonl':
                df = pd.read_json(path, lines=True)

            elif suffix == '.json':
                df = pd.read_json(path)

            else:
                return None

            return df

        except Exception as exc:
            print(f'Skipped {path.name}: {exc}')
            return None

    def merge(self) -> None:
        '''
        Merge datasets, preprocess them, and save result.
        '''
        dfs: List[pd.DataFrame] = []

        for file in self.input_dir.iterdir():
            if file.is_file():
                df = self.load_file(file)
                if df is not None:
                    print('Loaded:', file.name)
                    dfs.append(df)

        if not dfs:
            raise RuntimeError('No valid dataset files found.')

        df_all = pd.concat(dfs, ignore_index=True)

        for col in self.text_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col].apply(self.preprocess_text)
                df_all = df_all[df_all[col] != '']
            else:
                print(f'Warning: column "{col}" not found in dataset.')

        df_all = df_all.drop_duplicates()

        self.save_jsonl(df_all)

    def save_jsonl(self, df: pd.DataFrame) -> None:
        '''
        Save DataFrame as JSONL.

        Args:
            df: DataFrame to save.
        '''
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                obj = row.dropna().to_dict()
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print('Saved:', self.output_path)


def main() -> None:
    '''
    CLI entry point.
    '''
    parser = argparse.ArgumentParser(
        description='Merge CSV, TSV, and JSONL datasets.'
    )
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text_cols', nargs='+', required=False,
                        help='List of column names to preprocess', default=[])

    args = parser.parse_args()

    merger = DatasetMerger(
        args.input_dir,
        args.output,
        args.text_cols,
    )
    merger.merge()


if __name__ == '__main__':
    main()
