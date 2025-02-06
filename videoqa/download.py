import yt_dlp
import os

def download_video_yt_dlp(youtube_url, download_dir, filename="downloaded_video.mp4"):
    """
    Faz download de um vídeo do YouTube utilizando yt_dlp.
    
    :param youtube_url: URL do vídeo no YouTube
    :param download_dir: Diretório onde o vídeo será salvo
    :param filename: Nome do arquivo de saída
    :return: Caminho completo do vídeo baixado
    """
    os.makedirs(download_dir, exist_ok=True)
    output_path = os.path.join(download_dir, filename)
    
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return output_path
