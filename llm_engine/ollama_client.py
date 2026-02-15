from ollama import Client
from llm_engine.base import BaseLLMProvider
from typing import Optional


class OllamaClient(BaseLLMProvider):
    '''
    Ollama-based LLM provider using the chat API.

    This provider sends prompts to a locally running Ollama
    server using role-based messages (system / user).
    '''

    def __init__(
        self,
        user_prompt_template: str,
        system_prompt: Optional[str] = None,
        model: str = "gemma3:4b-it-qat",
    ):
        '''
        Initialize the Ollama client.

        Parameters
        ----------
        user_prompt_template : str
            Template for user prompts.
        system_prompt : str, optional
            System-level instruction defining model behavior.
        model : str, default="gemma3:4b-it-qat"
            Name of the Ollama model to use.
        '''
        super().__init__(
            user_prompt_template=user_prompt_template,
            system_prompt=system_prompt,
        )
        self.client = Client()
        self.model = model

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0
    ) -> str:
        '''
        Generate a response from the Ollama model using chat API.

        Parameters
        ----------
        user_prompt : str
            Rendered user prompt.
        system_prompt : str, optional
            System instruction passed to the model.

        Returns
        -------
        str
            Raw text response from the model.
        '''
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": user_prompt
        })

        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={"temperature": temperature}
        )

        return response["message"]["content"]
