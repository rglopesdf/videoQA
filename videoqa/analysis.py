import time
import json

def run_experiments(video_inspector, base64_frames, video_time_limit, pipeline="A", runs=1):
    """
    Executa múltiplas rodadas (runs) de inspeção do vídeo (pipeline A ou B) e coleta estatísticas.
    """
    results = []
    for i in range(runs):
        start = time.time()
        if pipeline.upper() == "A":
            result = video_inspector.inspect_pipeline_a(base64_frames, video_time_limit)
        elif pipeline.upper() == "B":
            result = video_inspector.inspect_pipeline_b(base64_frames, video_time_limit)
        else:
            raise ValueError("O pipeline deve ser 'A' ou 'B'")

        end = time.time()
        result["experiment_run"] = i + 1
        result["experiment_time"] = end - start
        results.append(result)
        print(f"Execução {i + 1} concluída em {end - start:.2f} segundos.")

    avg_time = sum(r["experiment_time"] for r in results) / runs

    return {
        "runs": results,
        "avg_time": avg_time
    }


def parse_final_inspection_json(final_inspection_str):
    """
    Exemplo de função para converter o texto retornado pela LLM 
    (que deve estar em formato JSON) em dicionário Python.
    """
    try:
        data = json.loads(final_inspection_str)
        return data
    except json.JSONDecodeError:
        return {}
