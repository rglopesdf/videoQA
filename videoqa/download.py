import os
import yt_dlp

def download_video_yt_dlp(youtube_url, download_dir, filename="downloaded_video.mp4"):
    """
    Faz download de um vídeo do YouTube utilizando yt_dlp, caso o arquivo não exista.
    
    :param youtube_url: URL do vídeo no YouTube
    :param download_dir: Diretório onde o vídeo será salvo
    :param filename: Nome do arquivo de saída
    :return: Caminho completo do vídeo baixado
    """
    os.makedirs(download_dir, exist_ok=True)
    output_path = os.path.join(download_dir, filename)
    
    # Verifica se o arquivo já existe
    if os.path.exists(output_path):
        print(f"Arquivo '{filename}' já existe em '{download_dir}'. Pulando download.")
        return output_path
    
    print("Iniciando download do vídeo...")
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return output_path
