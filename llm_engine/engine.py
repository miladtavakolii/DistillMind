from typing import List, Dict
from llm_engine.base import BaseLLMProvider


class LLMEngine:
    '''
    High-level orchestration engine for LLM providers.

    This engine:
        - Manages multiple providers (API keys or backends)
        - Automatically switches providers when one fails
        - Provides a unified interface for LLM generation
    '''

    def __init__(
        self,
        providers: List[BaseLLMProvider],
        temperature: float = 0.0,
    ) -> None:
        '''
        Initialize engine.

        Parameters
        ----------
        providers:
            List of LLM providers (can differ by API keys or models).

        temperature:
            Sampling temperature used during generation.
        '''
        if not providers:
            raise ValueError('At least one provider is required.')

        self.providers = providers
        self.temperature = temperature
        self.current_provider_idx = 0


    @property
    def provider(self) -> BaseLLMProvider:
        '''Return current active provider.'''
        return self.providers[self.current_provider_idx]


    def rotate_provider(self) -> None:
        '''
        Switch to next provider.

        Raises
        ------
        RuntimeError:
            If all providers fail.
        '''
        self.current_provider_idx += 1

        if self.current_provider_idx >= len(self.providers):
            raise RuntimeError('All LLM providers failed.')

        print(f'[Engine] Switched to provider #{self.current_provider_idx}')


    def generate(self, **fields: Dict) -> str:
        '''
        Generate structured output using the active provider.

        Parameters
        ----------
        **fields:
            Fields used to fill prompt templates
            (e.g., text, seed, instructions).

        Returns
        -------
        str
            raw text output returned by provider.
        '''
        while True:
            try:
                return self.provider.execute(
                    temperature=self.temperature,
                    **fields,
                )

            except Exception as exc:
                print('[Engine] Provider failed:', exc)
                self.rotate_provider()
