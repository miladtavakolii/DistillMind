from llm_engine.base import BaseLLMProvider


class LLMEngine:
    '''
    High-level sentiment analysis orchestrator.

    This class serves as a unified interface that delegates the actual
    LLM inference to a provider implementing `BaseSentimentProvider`.
    It is model-agnostic and can work with Ollama, Gemini, OpenAI, etc.

    Attributes
    ----------
    provider:
        The backend LLM provider responsible for analyzing text.
    '''

    def __init__(self, provider: BaseLLMProvider, temperature: float = 0):
        '''
        Initialize the sentiment engine.

        Parameters
        ----------
        provider:
            An instance of a provider capable of generating and parsing
            sentiment results.
        '''
        self.temperature = temperature
        self.provider = provider

    def analyze(self, text: str) -> dict[str, str]:
        '''
        Run full analysis on a new.

        This method passes normalized article metadata to the underlying
        provider and returns the parsed JSON sentiment result.

        Parameters
        ----------
        title: news text.

        Returns:
        Structured sentiment analysis result.
        '''
        return self.provider.analyze(
            temperature=self.temperature,
            text=text,
        )
