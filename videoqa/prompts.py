class PromptManager:
    def __init__(self, dense_prompt, answer_prompt, inspection_prompt, rewrite_prompt, few_shot_cnaes=""):
        """
        :param dense_prompt: Template para gerar descrições de frames.
        :param answer_prompt: Template para prompt de resposta final.
        :param inspection_prompt: Template para o prompt final de inspeção.
        :param rewrite_prompt: Template para reescrever a inspeção (Pipeline B ou pós-batch).
        :param few_shot_cnaes: Texto contendo a lista dos CNAES 2.0 (com descrições e observações) a ser incluído como few shot.
        """
        self.dense_prompt = dense_prompt
        self.answer_prompt = answer_prompt
        self.inspection_prompt = inspection_prompt
        self.rewrite_prompt = rewrite_prompt
        self.few_shot_cnaes = few_shot_cnaes

    def get_dense_prompt(self, frame_base64, frame_seq):
        """
        Retorna o prompt para gerar a descrição densa de um único frame.
        """
        return self.dense_prompt.format(frame_base64=frame_base64, frame_seq=frame_seq)

    def get_answer_prompt(self, descriptions):
        return self.answer_prompt.format(descriptions=descriptions)

    def get_inspection_prompt(self, descriptions):
        """
        Retorna o prompt de inspeção combinando as descrições dos frames e o few shot dos CNAES.
        Essa função é utilizada no Pipeline B para enviar o prompt consolidado.
        """
        message_parts = [
            f"Lista de CNAES 2.0:\n{self.few_shot_cnaes}",
            f"Descrições das imagens:\n{descriptions}",
            self.inspection_prompt
        ]
        message_content = "\n\n".join(message_parts)
        return message_content

    def get_inspection_messages(self, base64_frames, batch_sequence, pipeline="A"):
        """
        Monta o prompt final para inspeção conforme o manual da OpenAI.
        Retorna lista de mensagens no formato Chat (OpenAI).
        """
        # Para o Pipeline A, incluímos o few shot dos CNAES no instruction
        if pipeline == 'A':
            instruction = f"{self.few_shot_cnaes}\n\n{self.inspection_prompt}"
        else:
            # No Pipeline B, o instruction pode ser o dense_prompt
            instruction = self.dense_prompt

        message_content = [instruction]
        message_content.append(f"batch_sequence = {batch_sequence}")
        # Adiciona cada frame como dict {"image": base64, "resize": 768}
        message_content += [{"image": frame, "resize": 768} for frame in base64_frames]

        return [{"role": "user", "content": message_content}]