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

def save_experiment_statistics(experiments_results, final_inspection, csv_filename, 
                               llm_model, batch_size, pipeline, runs, youtube_url):
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
        "reasoning"
    ]

    file_exists = os.path.exists(csv_filename)

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Itera sobre cada execução (run)
        for run in experiments_results.get("runs", []):
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
                "reasoning": final_inspection.get("reasoning")
            })

    print(f"\nEstatísticas dos experimentos foram salvas (ou atualizadas) no arquivo '{csv_filename}'.")
    df = pd.read_csv(csv_filename)
    return df
