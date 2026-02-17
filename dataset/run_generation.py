import os
from pathlib import Path
from typing import List, Type
import argparse
import json

from llm_engine.engine import LLMEngine
from llm_engine.base import BaseLLMProvider
from llm_engine.ollama_client import OllamaClient
from llm_engine.gemini_client import GeminiClient
from dataset.llm_data_generator import LLMDataGenerator
from utils.prompt_utils import load_prompt


class DatasetGenerationManager:
    '''
    High-level manager for generating datasets using LLMs.

    Responsibilities:
        - Load system/user prompts from files
        - Build providers
        - Initialize LLMEngine
        - Generate and save dataset
    '''

    def __init__(
        self,
        phase: str,
        provider_class: Type[BaseLLMProvider],
        providers_config_path: str,
        output_dir: str,
        temperature: float = 0.7,
        batch_size: int = 100,
    ):
        '''
        Initialize the manager.

        Parameters
        ----------
        phase : str
            Phase name (e.g., 'phase2', 'phase3') to select prompt files.
        provider_class : Type[BaseLLMProvider]
            LLM provider class to instantiate (e.g., OllamaClient, GeminiClient)
        api_keys : List[Tuple[str, str]]
            List of model names and API keys to create multiple providers.
        output_dir : str
            Directory to save generated dataset.
        temperature : float
            Temperature for text generation.
        batch_size : int
            Number of samples per batch.
        '''
        self.phase = phase
        self.provider_class = provider_class
        self.providers_config_path = providers_config_path
        self.output_dir = output_dir
        self.temperature = temperature
        self.batch_size = batch_size

        # Load prompts
        self.system_prompt = load_prompt(
            os.path.join('dataset', 'prompts', phase, 'system.txt')
        )
        self.user_prompt = load_prompt(
            os.path.join('dataset', 'prompts', phase, 'user.txt')
        )

        # Build providers
        self.providers = self._build_providers()
        self.engine = LLMEngine(providers=self.providers, temperature=self.temperature)
        self.generator = LLMDataGenerator(self.engine)

    def _build_providers(self) -> List[BaseLLMProvider]:
        '''Instantiate providers from API keys.'''
        if not os.path.exists(self.providers_config_path):
            raise FileNotFoundError(f"Providers config not found: {self.providers_config_path}")

        with open(self.providers_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        providers = []
        for entry in config:
            model_name = entry.get("model")
            api_key = entry.get("api_key")
            if not model_name or not api_key:
                continue
            provider = self.provider_class(
                user_prompt_template=self.user_prompt,
                system_prompt=self.system_prompt,
                api_key=api_key,
                model=model_name
            )
            providers.append(provider)
        if not providers:
            raise ValueError("No valid providers found in config file.")
        return providers

    def generate_dataset(self, total_samples: int) -> None:
        '''
        Generate the dataset and save to output_dir.

        Parameters
        ----------
        total_samples : int
            Total number of samples to generate.
        '''
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.generator.generate_dataset(
            output_dir=self.output_dir,
            total_samples=total_samples,
            batch_size=self.batch_size
        )


def main():
    parser = argparse.ArgumentParser(description='LLM Dataset Generator')

    parser.add_argument('--phase', type=str, required=True,
                        help='Phase name (phase2, phase3) to select prompt files')
    parser.add_argument('--provider', type=str, required=True,
                        help='Provider name (Gemini, Ollama)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated dataset')
    parser.add_argument('--total_samples', type=int, required=True,
                        help='Total number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of samples per batch')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--providers_config', type=str, required=True,
                        help='Path to JSON file with API keys and models')


    args = parser.parse_args()

    manager = DatasetGenerationManager(
        phase=args.phase,
        provider_class=OllamaClient if args.provider == 'ollama' else GeminiClient,
        providers_config_path=args.providers_config,
        output_dir=args.output_dir,
        temperature=args.temperature,
        batch_size=args.batch_size
    )

    manager.generate_dataset(total_samples=args.total_samples)


if __name__ == '__main__':
    main()  
