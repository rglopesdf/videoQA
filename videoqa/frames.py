import cv2
import base64

def extract_base64_frames(video_path, frame_interval=None):
    """
    Abre o vídeo (video_path), extrai frames (a cada frame_interval) e converte-os para base64.

    :param video_path: Caminho do arquivo de vídeo
    :param frame_interval: Intervalo de frames (se None, default = 1 frame por segundo)
    :return: (base64Frames, metadata)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Por padrão, se frame_interval não for informado, usamos 1 frame por segundo
    if frame_interval is None:
        frame_interval = int(fps)

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
    return base64_frames, metadata
