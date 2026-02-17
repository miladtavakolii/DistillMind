import json
import os
from typing import Dict, List, Optional

from llm_engine.engine import LLMEngine
from utils.json_utils import extract_json


class LLMDataGenerator:
    '''
    Dataset generator using LLMEngine.

    This class:
        - Loads prompt templates from files
        - Requests data generation from LLM
        - Aggregates results
        - Saves dataset to disk

    Designed for Phase 2 and Phase 3 dataset creation.
    '''

    def __init__(
        self,
        engine: LLMEngine,
    ) -> None:
        '''
        Initialize generator.

        Parameters
        ----------
        engine:
            LLMEngine used for generation.
        '''
        self.engine = engine

    def generate_batch(self, **fields) -> str:
        '''
        Generate one batch of samples.

        Parameters
        ----------
        **fields:
            Values used to fill prompt template
            (e.g., n_samples=100, seed=42)

        Returns
        -------
        str
            Generated samples.
        '''
        prompt_fields = dict(fields)
        raw_output = self.engine.generate(**prompt_fields)

        return raw_output


    def generate_dataset(
        self,
        output_dir: str,
        total_samples: int,
        batch_size: int,
        **fields,
    ) -> None:
        '''
        Generate dataset and save to disk.

        Parameters
        ----------
        output_dir:
            Output dataset JSON directory.

        total_samples:
            Total number of samples required.

        batch_size:
            Number of samples requested per batch.

        **fields:
            Prompt fields passed to model.
        '''

        os.makedirs(output_dir, exist_ok=True)

        file_index = len(os.listdir(output_dir))
        generated = 0

        while generated < total_samples:
            remaining = total_samples - generated
            current_batch = min(batch_size, remaining)

            print(f"[Generator] Generated {generated}/{total_samples}")

            batch = self.generate_batch(
                **fields,
                n_samples=current_batch,
            )
            
            exrtacted_json = extract_json(batch)

            file_path = os.path.join(
                output_dir,
                f"batch_{file_index:05d}.json",
            )

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(exrtacted_json, f, ensure_ascii=False, indent=2)

            generated += len(exrtacted_json)
            file_index += 1

        print("[Generator] Generation completed.")
