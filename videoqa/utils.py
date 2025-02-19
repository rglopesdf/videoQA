import textwrap
import base64
import csv
import os
import pandas as pd
from IPython.display import display, Image

def format_text(text, width=80):
    """
    Formata um texto longo para quebrar linhas automaticamente respeitando a largura especificada.
    """
    return textwrap.fill(text, width=width)

def show_inspection_images(llm_response, video_frames, frames_per_batch):
    """
    Exibe as imagens relevantes indicadas na resposta da LLM.
    Formato esperado em llm_response["images"]: [[frame_index, batch_sequence], ...]
    """
    imagens = llm_response.get("images", [])
    if not imagens:
        print("Nenhuma imagem indicada na resposta.")
        return

    for pair in imagens:
        try:
            frame_index, batch_sequence = pair
            global_index = batch_sequence * frames_per_batch + frame_index
            if 0 <= global_index < len(video_frames):
                frame_base64 = video_frames[global_index]
                img_bytes = base64.b64decode(frame_base64)
                display(Image(data=img_bytes))
            else:
                print(f"Índice global {global_index} fora dos limites do vetor de frames.")
        except Exception as e:
            print("Erro ao processar o par", pair, ":", e)

import os
import csv
import json
import pandas as pd

def save_experiment_statistics(experiments_results, csv_filename, 
                               llm_model, batch_size, pipeline, runs, youtube_url, useFewshot = False ):
    """
    Salva (ou acrescenta) as estatísticas dos experimentos em um arquivo CSV.
    Retorna um DataFrame com as estatísticas consolidadas.
    """
    fieldnames = [
        "youtube_url",
        "experiment_run",
        "llmModel",
        "batch_size",
        "pipeline",
        "runs",
        "total_tokens",
        "total_time",
        "experiment_time",
        "cnae",
        "cnae_divisao",
        "cnae_divisao_descricao",
        "cnae_grupo",
        "cnae_grupo_descricao",
        "cnae_classe",
        "cnae_classe_descricao",
        "cnae_subclasse",
        "cnae_subclasse_descricao",
        "reasoning",
        "useFewshot"
    ]

    file_exists = os.path.exists(csv_filename)

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Itera sobre cada execução (run)
        for run in experiments_results.get("runs", []):
            final_inspection_data = run.get("final_inspection", "{}")

            # Se for uma string, tenta converter usando json.loads()
            if isinstance(final_inspection_data, str):
                try:
                    final_inspection = json.loads(final_inspection_data)
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON na execução {run.get('experiment_run')}")
                    final_inspection = {}
            # Se for uma lista, tenta pegar o primeiro elemento (convertendo-o se necessário)
            elif isinstance(final_inspection_data, list):
                try:
                    elem = final_inspection_data[0]
                    if isinstance(elem, str):
                        final_inspection = json.loads(elem)
                    else:
                        final_inspection = elem
                except (IndexError, json.JSONDecodeError):
                    print(f"Erro ao processar `final_inspection` na execução {run.get('experiment_run')}")
                    final_inspection = {}
            else:
                final_inspection = final_inspection_data

            # Se, após o processamento, final_inspection ainda for uma lista,
            # pega o primeiro elemento (ou usa {} se estiver vazia)
            if isinstance(final_inspection, list):
                final_inspection = final_inspection[0] if final_inspection else {}

            writer.writerow({
                "youtube_url": youtube_url,
                "experiment_run": run.get("experiment_run"),
                "llmModel": llm_model,
                "batch_size": batch_size,
                "pipeline": pipeline,
                "runs": runs,
                "total_tokens": run.get("total_tokens"),
                "total_time": run.get("total_time"),
                "experiment_time": run.get("experiment_time"),
                "cnae": final_inspection.get("cnae"),
                "cnae_divisao": final_inspection.get("cnae_divisao"),
                "cnae_divisao_descricao": final_inspection.get("cnae_divisao_descricao"),
                "cnae_grupo": final_inspection.get("cnae_grupo"),
                "cnae_grupo_descricao": final_inspection.get("cnae_grupo_descricao"),
                "cnae_classe": final_inspection.get("cnae_classe"),
                "cnae_classe_descricao": final_inspection.get("cnae_classe_descricao"),
                "cnae_subclasse": final_inspection.get("cnae_subclasse"),
                "cnae_subclasse_descricao": final_inspection.get("cnae_subclasse_descricao"),
                "reasoning": final_inspection.get("reasoning"),
                "useFewshot": useFewshot
            })

    print(f"\nEstatísticas dos experimentos foram salvas (ou atualizadas) no arquivo '{csv_filename}'.")
    df = pd.read_csv(csv_filename)
    return df

import os
import json
import csv
import pandas as pd

def save_experiment_statistics_qa(experiments_results, csv_filename, 
                               llm_model, batch_size, pipeline, runs, youtube_url, useFewshot=False):
    """
    Salva (ou acrescenta) as estatísticas dos experimentos em um arquivo CSV, considerando a nova estrutura
    baseada em grupos de perguntas, onde cada resposta possui:
        - group_number: número do agrupamento de perguntas
        - question_number: número da pergunta
        - question_answer: resposta da questão
        - reasoning: texto explicando sucintamente a justificativa
        - images: lista de [frame_index, batch_sequence]
        
    Cada linha do CSV corresponderá a uma resposta individual, mantendo os metadados do experimento.
    
    Parâmetros:
        experiments_results (dict): Resultados do experimento, contendo uma chave "runs" com os dados de cada execução.
        csv_filename (str): Nome do arquivo CSV onde os dados serão salvos.
        llm_model (str): Nome/modelo do LLM utilizado.
        batch_size (int): Tamanho do lote processado.
        pipeline (str): Nome ou descrição do pipeline utilizado.
        runs (int): Número total de execuções do experimento.
        youtube_url (str): URL do vídeo do YouTube associado ao experimento.
        useFewshot (bool, opcional): Indica se foi utilizado o método Fewshot. Padrão é False.
    
    Retorna:
        pd.DataFrame: DataFrame contendo as estatísticas consolidadas.
    """
    # Definição dos nomes dos campos (colunas) que serão salvos no CSV
    fieldnames = [
        "youtube_url",
        "experiment_run",
        "llmModel",
        "batch_size",
        "pipeline",
        "runs",
        "total_tokens",
        "total_time",
        "experiment_time",
        "group_number",
        "question_number",
        "question_answer",
        "reasoning",
        "images",
        "useFewshot"
    ]

    # Verifica se o arquivo CSV já existe para não sobrescrever o cabeçalho
    file_exists = os.path.exists(csv_filename)

    # Abre o arquivo CSV em modo de acréscimo
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Itera sobre cada execução (run) do experimento
        for run in experiments_results.get("runs", []):
            # Recupera o conteúdo referente à inspeção final (respostas das perguntas)
            final_inspection_data = run.get("final_inspection", "{}")

            # Processamento do campo 'final_inspection' conforme o tipo:
            # - Se for string, tenta converter para objeto JSON
            # - Se for lista, processa cada item (que pode estar em formato string ou dict)
            # - Se for dict, encapsula em uma lista para padronização
            if isinstance(final_inspection_data, str):
                try:
                    final_inspection = json.loads(final_inspection_data)
                    # Se o resultado for um dicionário, encapsula-o em uma lista
                    if isinstance(final_inspection, dict):
                        final_inspection = [final_inspection]
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON na execução {run.get('experiment_run')}")
                    final_inspection = []
            elif isinstance(final_inspection_data, list):
                final_inspection = []
                for item in final_inspection_data:
                    if isinstance(item, str):
                        try:
                            parsed_item = json.loads(item)
                            # Se o item convertido for uma lista, estende a lista principal
                            if isinstance(parsed_item, list):
                                final_inspection.extend(parsed_item)
                            else:
                                final_inspection.append(parsed_item)
                        except json.JSONDecodeError:
                            print(f"Erro ao processar item em final_inspection na execução {run.get('experiment_run')}")
                    elif isinstance(item, dict):
                        final_inspection.append(item)
                    else:
                        print(f"Tipo inesperado em final_inspection na execução {run.get('experiment_run')}")
            elif isinstance(final_inspection_data, dict):
                final_inspection = [final_inspection_data]
            else:
                final_inspection = []

            # Para cada resposta individual (cada questão) presente na inspeção final,
            # escreve uma linha no CSV com os metadados do experimento e os dados da resposta.
            for answer in final_inspection:
                writer.writerow({
                    "youtube_url": youtube_url,
                    "experiment_run": run.get("experiment_run"),
                    "llmModel": llm_model,
                    "batch_size": batch_size,
                    "pipeline": pipeline,
                    "runs": runs,
                    "total_tokens": run.get("total_tokens"),
                    "total_time": run.get("total_time"),
                    "experiment_time": run.get("experiment_time"),
                    "group_number": answer.get("group_number"),
                    "question_number": answer.get("question_number"),
                    "question_answer": answer.get("question_answer"),
                    "reasoning": answer.get("reasoning"),
                    "images": answer.get("images"),
                    "useFewshot": useFewshot
                })

    print(f"\nEstatísticas dos experimentos foram salvas (ou atualizadas) no arquivo '{csv_filename}'.")
    # Carrega o CSV em um DataFrame e o retorna
    df = pd.read_csv(csv_filename)
    return df
