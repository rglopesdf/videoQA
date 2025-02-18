import os
import json

from videoqa.download import download_video_yt_dlp
from videoqa.frames import extract_base64_frames
from videoqa.llms import OpenAI_LLM
from videoqa.prompts import PromptManager
from videoqa.inspector import VideoProcessor, VideoInspector
from videoqa.analysis import run_experiments, parse_final_inspection_json
from videoqa.utils import format_text, save_experiment_statistics

def main():
    # Exemplo de uso básico via script
    youtube_url = "https://www.youtube.com/watch?v=SEU_VIDEO_ID"
    download_dir = "./downloads"
    video_filename = "meu_video.mp4"

    print("Baixando vídeo...")
    video_path = download_video_yt_dlp(youtube_url, download_dir, filename=video_filename)
    print("Vídeo salvo em:", video_path)
    
    print("Extraindo frames...")
    processor = VideoProcessor(video_path)
    frames_list, metadata = processor.process_video()
    print("Metadados:", metadata)

    # Configura prompts
    json_mask = """
    {
      "cnae": "...",
      "cnae_divisao": "...",
      "cnae_divisao_descricao": "...",
      "cnae_grupo": "...",
      "cnae_grupo_descricao": "...",
      "cnae_classe": "...",
      "cnae_classe_descricao": "...",
      "cnae_subclasse": "...",
      "cnae_subclasse_descricao": "...",
      "reasoning": "...",
      "images": [
        [frame_index, batch_sequence]
      ]
    }
    """

    prompt_manager = PromptManager(
        dense_prompt=(
            "Você é um especialista em análise visual e deve descrever os frames..."
        ),
        answer_prompt=(
            "Com base nas descrições: {descriptions}, retorne o CNAE neste formato JSON: " + json_mask
        ),
        inspection_prompt=(
            "Pergunta: qual o CNAE? Responda no formato JSON a seguir: " + json_mask
        ),
        rewrite_prompt=(
            "Você recebeu diversas respostas parciais: {inspections}. Consolidar em um só JSON..."
        )
    )

    # Configura LLM (OpenAI)
    api_key = os.getenv("OPENAI_API_KEY", "SUA_CHAVE_AQUI")
    llm = OpenAI_LLM(api_key=api_key, model="gpt-4")

    video_inspector = VideoInspector(llm, prompt_manager, batch_size=10)
    video_inspector.register_correct_cnae("1234-5/67")  # Exemplo

    # Executa experimentos
    experiments_results = run_experiments(
        video_inspector,
        frames_list,
        video_time_limit=30,
        pipeline="A",
        runs=2
    )

    # Exibe resultados
    for idx, run in enumerate(experiments_results["runs"]):
        final_inspection_str = run.get("final_inspection", "{}")
        final_inspection = parse_final_inspection_json(final_inspection_str)
        reasoning = final_inspection.get("reasoning", "")
        print(f"\n--- Execução {idx+1} ---")
        print("CNAE:", final_inspection.get("cnae"))
        print("Reasoning:\n", format_text(reasoning, width=80))

    # Salva estatísticas em CSV
    csv_filename = "experiment_statistics.csv"
    final_inspection = parse_final_inspection_json(
        experiments_results["runs"][-1].get("final_inspection", "{}")
    )
    df = save_experiment_statistics(
        experiments_results,
        final_inspection,
        csv_filename,
        llm_model="gpt-4",
        batch_size=10,
        pipeline="A",
        runs=2,
        youtube_url=youtube_url
    )
    print(df.head())

if __name__ == "__main__":
    main()
