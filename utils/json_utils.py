import json
import re

def extract_json(self, text: str) -> dict:
    '''
    Extract and parse a valid JSON object from arbitrary LLM output.

    This method:
        - Removes markdown code fences (```json ... ```)
        - Normalizes non-standard quotes
        - Attempts full-text JSON parsing
        - Falls back to regex-based JSON extraction when needed

    Parameters
    ----------
    text: Raw output returned by the LLM.

    Returns
    -------
    Parsed JSON object extracted from the model output.

    Raises
    ------
    ValueError:
        If no valid JSON structure can be recovered.
    '''
    text = text.replace('”', '\'')
    # Remove markdown code fences
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Try parse as JSON directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If JSON is inside text somewhere:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f'[ERROR] No valid JSON found in model output:\n {text}')
