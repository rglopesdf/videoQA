import time
import openai
import google.generativeai as genai

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
            #temperature=0.0
        )
        end_time = time.time()

        output_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        token_usage = response.usage
        return {
            "output": output_text,
            "token_usage": token_usage,
            "processing_time": end_time - start_time
        }

class Gemini_LLM(LLMBase):
    def __init__(self, api_key, llmmodel, config):
        super().__init__("Gemini")
        genai.configure(api_key=api_key)
        self.config = config
        self.llmmodel = llmmodel


    def run_prompt(self, prompt):

        start_time = time.time()  

        modelo = genai.GenerativeModel(
          model_name=self.llmmodel,
          generation_config=self.config,
        )

        session_chat = modelo.start_chat(history = prompt)      
        response = session_chat.send_message("INSERT_INPUT_HERE")
        end_time = time.time()

        #output_text = response.text
        output_text = response.text.replace("```json", "").replace("```", "").strip()
        token_usage = response.usage_metadata
        return {
            "output": output_text,
            "token_usage": response.usage_metadata,
            "processing_time": end_time - start_time
        }

        return 

    def upload_para_gemini(caminho, mime_type=None):
        """
        Faz o upload do arquivo para o Gemini.
        
        Veja: https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        arquivo = genai.upload_file(caminho, mime_type=mime_type)
        print(f"Arquivo '{arquivo.display_name}' enviado como: {arquivo.uri}")
        return arquivo

    def aguardar_arquivos_ativos(arquivos):
        """
        Aguarda que os arquivos enviados estejam ativos.
        
        Alguns arquivos precisam ser processados antes de serem utilizados como entrada.
        O status pode ser verificado consultando o campo "state" do arquivo.
        
        Essa implementação usa um loop de polling simples; em produção, recomenda-se
        uma abordagem mais sofisticada.
        """
        print("Aguardando o processamento dos arquivos...")
        for nome in (arquivo.name for arquivo in arquivos):
            arquivo = genai.get_file(nome)
            while arquivo.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                arquivo = genai.get_file(nome)
            if arquivo.state.name != "ACTIVE":
                raise Exception(f"O arquivo {arquivo.name} falhou no processamento.")
        print("...todos os arquivos estão prontos\n")


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
