import cv2
import base64
import json

def extract_base64_frames(video_path, frame_interval=None, cache_json_path=None):
    """
    Abre o vídeo (video_path), extrai frames (a cada frame_interval) e converte-os para base64.
    Caso 'cache_json_path' seja fornecido e o arquivo exista, carrega os frames e metadados dali.
    
    :param video_path: Caminho do arquivo de vídeo.
    :param frame_interval: Intervalo de frames (se None, default = 1 frame por segundo).
    :param cache_json_path: Caminho para arquivo JSON de cache. Se existir, carregamos direto.
    :return: (base64_frames, metadata)
    """
    # Se especificaram um cache e ele já existe, apenas carrega o conteúdo
    if cache_json_path and os.path.exists(cache_json_path):
        print(f"Arquivo de cache '{cache_json_path}' encontrado. Carregando frames do cache...")
        with open(cache_json_path, 'r') as f:
            data = json.load(f)
        return data["base64_frames"], data["metadata"]
    
    # Caso não exista cache ou não tenha sido especificado, faz a extração
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Por padrão, se frame_interval não for informado, usar 1 frame por segundo
    if frame_interval is None:
        frame_interval = int(fps) if fps else 1

    base64_frames = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            success_encode, buffer = cv2.imencode(".jpg", frame)
            if success_encode:
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                base64_frames.append(frame_b64)
        frame_count += 1
    
    cap.release()

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "frame_interval": frame_interval
    }

    print(f"{len(base64_frames)} frames extraídos (a cada {frame_interval} frames).")

    # Se um caminho de cache foi informado, salva nele
    if cache_json_path:
        os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)
        with open(cache_json_path, 'w') as f:
            json.dump({"base64_frames": base64_frames, "metadata": metadata}, f)
        print(f"Frames e metadados salvos no cache '{cache_json_path}'.")

    return base64_frames, metadata
