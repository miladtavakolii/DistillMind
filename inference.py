import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from typing import Optional


class GemmaChatInference:
    '''
    A lightweight inference wrapper for a fine-tuned Gemma chat model.

    This class loads a local HuggingFace-compatible causal language model
    and provides a chat-style generation interface using the tokenizer's
    chat template.

    Parameters
    ----------
    model_path : str
        Path to the locally saved (merged) model directory.
    device : Optional[str]
        Device to run inference on ('cuda' or 'cpu').
        If None, automatically selects CUDA if available.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling parameter.
    top_k : int
        Top-k sampling parameter.
    '''

    def __init__(
        self,
        model_path: str = './model',
        device: Optional[str] = None,
        max_new_tokens: int = 125,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> None:

        self.model_path = model_path
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        print('Loading model...')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
        )

        self.model.eval()

        print(f'Model loaded successfully on {self.device}.\n')

    # --------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------

    def _build_input_text(self, system_prompt: str, user_prompt: str) -> str:
        '''
        Build chat-formatted input using tokenizer chat template.
        '''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).removeprefix('<bos>')

        return text

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = True,
    ) -> Optional[str]:
        '''
        Generate a response from the model.

        Parameters
        ----------
        system_prompt : str
            Instruction or system-level prompt.
        user_prompt : str
            User input text.
        stream : bool
            If True, streams output token-by-token to stdout.
            If False, returns the full generated string.

        Returns
        -------
        Optional[str]
            Returns generated text if stream=False.
            Otherwise prints tokens and returns None.
        '''

        text = self._build_input_text(system_prompt, user_prompt)

        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            if stream:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True)

                self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    streamer=streamer,
                )
                return None
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )

                decoded = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                )

                return decoded


if __name__ == '__main__':

    system_prompt = (
        'You are a Persian mental-state classifier. '
        'Given a Persian text, first explain briefly why it belongs '
        'to a category, then output the final label. '
        'Possible labels: fear, withdrawal, other. '
        'Return your answer in JSON with keys: reason, label.'
    )

    model = GemmaChatInference(model_path='./model')

    print('Chat ready. Type "exit" to quit.\n')

    while True:
        user_input = input('>> ')

        if user_input.lower() == 'exit':
            break

        model.generate(system_prompt, user_input, stream=True)
        print()
