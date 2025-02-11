import time
import openai

class LLMBase:
    def __init__(self, name):
        self.name = name

    def run_prompt(self, prompt):
        raise NotImplementedError("Este método deve ser implementado pela subclasse.")

class OpenAI_LLM(LLMBase):
    def __init__(self, api_key, model="gpt-4"):
        """
        Parâmetros:
          - api_key: Chave da API da OpenAI.
          - model: Nome do modelo (ex.: "gpt-4").
        """
        super().__init__("OpenAI")
        openai.api_key = api_key
        self.model = model

    def run_prompt(self, prompt):
        """
        Se prompt for uma lista (de mensagens) já está formatado conforme a OpenAI (chat).
        Caso contrário, converte para o formato de lista.
        """
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        end_time = time.time()

        output_text = response.choices[0].message.content
        token_usage = response.usage
        return {
            "output": output_text,
            "token_usage": token_usage,
            "processing_time": end_time - start_time
        }

class Gemini_LLM(LLMBase):
    def __init__(self, config):
        super().__init__("Gemini")
        self.config = config

    def run_prompt(self, prompt):
        time.sleep(0.5)
        return {
            "output": "Resposta simulada pelo Gemini.",
            "token_usage": {"total_tokens": 100},
            "processing_time": 0.5
        }

class Llama32b_LLM(LLMBase):
    def __init__(self, config):
        super().__init__("Llama32b")
        self.config = config

    def run_prompt(self, prompt):
        time.sleep(0.5)
        return {
            "output": "Resposta simulada pelo Llama32b.",
            "token_usage": {"total_tokens": 120},
            "processing_time": 0.5
        }
