class PromptManager:
    def __init__(self, dense_prompt, answer_prompt, inspection_prompt, rewrite_prompt):
        """
        :param dense_prompt: Template para gerar descrições de frames.
        :param answer_prompt: Template para prompt de resposta final.
        :param inspection_prompt: Template para o prompt final de inspeção.
        :param rewrite_prompt: Template para reescrever a inspeção (Pipeline B ou pós-batch).
        """
        self.dense_prompt = dense_prompt
        self.answer_prompt = answer_prompt
        self.inspection_prompt = inspection_prompt
        self.rewrite_prompt = rewrite_prompt

    def get_dense_prompt(self, frame_base64, frame_seq):
        return self.dense_prompt.format(frame_base64=frame_base64, frame_seq=frame_seq)

    def get_answer_prompt(self, descriptions):
        return self.answer_prompt.format(descriptions=descriptions)

    def get_inspection_prompt(self, descriptions):
        """
        Retorna o prompt de inspeção final combinando as descrições (texto) dos frames.
        """
        safe_descriptions = descriptions.replace("{", "{{").replace("}", "}}")
        return self.inspection_prompt.format(descriptions=safe_descriptions)
        # return self.inspection_prompt.format(descriptions=descriptions)

    def get_inspection_messages(self, base64_frames, batch_sequence, pipeline="A"):
        """
        Monta o prompt final para inspeção conforme o manual da OpenAI.
        Retorna lista de mensagens no formato Chat (OpenAI).
        """
        if pipeline == 'A':
            # Usa inspection_prompt como 'instruction'
            instruction = self.inspection_prompt
        else:
            # Exemplo: pode usar dense_prompt como 'instruction'
            instruction = self.dense_prompt

        message_content = [instruction]
        message_content.append(f"batch_sequence = {batch_sequence}")
        # Adiciona cada frame como dict {"image": base64, "resize": 768} 
        message_content += [{"image": frame, "resize": 768} for frame in base64_frames]

        return [{"role": "user", "content": message_content}]
