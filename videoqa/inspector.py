import json
import time
from .frames import extract_base64_frames

class VideoProcessor:
    def __init__(self, video_path):
        """
        :param video_path: Caminho do vídeo
        """
        self.video_path = video_path

    def process_video(self, frame_interval=None):
        """
        Extrai os frames do vídeo em formato base64.
        Retorna (base64Frames, metadata).
        """
        return extract_base64_frames(self.video_path, frame_interval=frame_interval)


class VideoInspector:
    def __init__(self, llm, prompt_manager, batch_size=20):
        """
        :param llm: Instância de LLMBase (ex: OpenAI_LLM).
        :param prompt_manager: Instância de PromptManager.
        :param batch_size: Número de frames por batch.
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.batch_size = batch_size
        self.correct_cnae = None

    def register_correct_cnae(self, correct_cnae):
        self.correct_cnae = correct_cnae

    def inspect_pipeline_a(self, base64_frames, video_time_limit):
        """
        Pipeline A: Envia frames em batches, e por fim reescreve numa resposta única.
        """
        batch_results = []
        total_tokens = 0
        total_time = 0
        video_time_current = 0
        batch_sequence = 0

        # Envia lotes de frames
        for i in range(0, len(base64_frames), self.batch_size):
            batch = base64_frames[i : i + self.batch_size]
            prompt_messages = self.prompt_manager.get_inspection_messages(
                batch, batch_sequence, pipeline='A'
            )
            result = self.llm.run_prompt(prompt_messages)
            batch_results.append(result["output"])
            total_tokens += result["token_usage"].total_tokens
            total_time += result["processing_time"]
            video_time_current += self.batch_size
            batch_sequence += 1

            if video_time_current >= video_time_limit:
                break

        combined_text = "\n".join(batch_results)

        # Reescrita final para consolidar
        rewrite_prompt_message = [{
            "role": "user",
            "content": self.prompt_manager.rewrite_prompt.format(inspections=combined_text)
        }]
        final_result = self.llm.run_prompt(rewrite_prompt_message)
        total_tokens += final_result["token_usage"].total_tokens
        total_time += final_result["processing_time"]

        return {
            "final_inspection": final_result["output"],
            "batch_inspections": batch_results,
            "total_tokens": total_tokens,
            "total_time": total_time
        }

    def inspect_pipeline_b(self, base64_frames, video_time_limit):
        """
        Pipeline B: Parecido com o A, mas ao final combina o texto em outro prompt 
        e envia novamente para a LLM (exemplo de variação).
        """
        batch_results = []
        total_tokens = 0
        total_time = 0
        video_time_current = 0
        batch_sequence = 0

        for i in range(0, len(base64_frames), self.batch_size):
            batch = base64_frames[i : i + self.batch_size]
            prompt_messages = self.prompt_manager.get_inspection_messages(
                batch, batch_sequence, pipeline='B'
            )
            result = self.llm.run_prompt(prompt_messages)
            batch_results.append(result["output"])

            total_tokens += result["token_usage"].total_tokens
            total_time += result["processing_time"]
            video_time_current += self.batch_size
            batch_sequence += 1

            if video_time_current >= video_time_limit:
                break

        combined_text = "\n".join(batch_results)
        # Envia ao prompt de inspeção final, por exemplo
        rewrite_prompt_message = [{
            "role": "user",
            "content": self.prompt_manager.get_inspection_prompt(descriptions=combined_text)
        }]
        final_result = self.llm.run_prompt(rewrite_prompt_message)
        total_tokens += final_result["token_usage"].total_tokens
        total_time += final_result["processing_time"]

        return {
            "final_inspection": final_result["output"],
            "batch_inspections": batch_results,
            "total_tokens": total_tokens,
            "total_time": total_time
        }

    def evaluate_inspection(self, predicted_cnae):
        """
        Exemplo de função de avaliação simples 
        (caso você queira verificar se a LLM acerta o CNAE).
        """
        overall = 1.0 if predicted_cnae == self.correct_cnae else 0.0
        return {"overall": overall}

class VideoInspectorGemini:
    def __init__(self, llm, arquivo, prompt_manager):
        """
        :param llm: Instância de LLMBase (ex: OpenAI_LLM).
        :param prompt_manager: Instância de PromptManager.
        :param batch_size: Número de frames por batch.
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.arquivo = arquivo
        self.correct_cnae = None

    def register_correct_cnae(self, correct_cnae):
        self.correct_cnae = correct_cnae

    def inspect_pipeline_a(self):
        """
        Pipeline A: Envia vídeo completo
        """
        total_tokens = 0
        total_time = 0

        json_mask = """
        {
          "cnae": "EX: 62.01-1-00",
          "cnae_divisao": "62",
          "cnae_divisao_descricao": "Descrição da divisao",
          "cnae_grupo": "01",
          "cnae_grupo_descricao": "Descrição do grupo",
          "cnae_classe": "1",
          "cnae_classe_descricao": "Descrição da classe",
          "cnae_subclasse": "00",
          "cnae_subclasse_descricao": "Descrição da subclasse",
          "reasoning": "Texto explicando como os elementos visuais indicam o código CNAE. Seja muito sucinto nesta explicação",
          "images": [
            [frame_index, batch_sequence],
            [frame_index, batch_sequence],
            ...
          ]
        }

        """
        # Envia lotes de frames

        prompt_messages=[
            {
                "role": "user",
                "parts": [
                    self.arquivo,
                ],
            },
            {
                "role": "user",
                "parts": [
                    (
                        "### Contexto ###\n"
                        "Você é um especialista em análise visual e sua tarefa é responder à pergunta abaixo, "
                        "com base nas imagens disponibilizadas.\n"
                        "Pergunta: Baseado nas imagens, estime qual é o código CNAE do empreendimento, segundo a classificação nacional de "
                        "atividades econômicas do Brasil?\n"
                        "Responda com a máscara completa: Divisão, Grupo, Classe e Subclasse (X.XX-XX-X). "
                        "Responda apenas UM CNAE. Se houver várias atividades, informe APENAS o CNAE predominante.\n"
                        "Informe a lista de imagens que justificam sua resposta. Seja muito sucinto, informando apenas "
                        "as imagens imprescindíveis para dar credibilidade à sua resposta. NUNCA apresente mais do que 10 imagens."
                      f"Lista de CNAEs: {self.prompt_manager.few_shot_cnaes}"
                      f"Retorne a resposta no seguinte formato JSON, sem delimitadores de bloco de código: {json_mask}"                    
                    )
                        ],
              }
                ]


        result = self.llm.run_prompt(prompt_messages)
        total_tokens = result["token_usage"].total_token_count
        total_time = result["processing_time"]


        return {
            "final_inspection": result["output"],
            "batch_inspections": result,
            "total_tokens": total_tokens,
            "total_time": total_time
        }
