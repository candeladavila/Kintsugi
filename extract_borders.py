import os
import cv2

def extract_borders(image_path: str, out_dir: str, border_width: int = 100) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    bw = min(border_width, h, w)

    top = img[0:bw, :].copy()
    bottom = img[h-bw:h, :].copy()
    left = img[:, 0:bw].copy()
    right = img[:, w-bw:w].copy()

    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{base}_top_{bw}px.png"), top)
    cv2.imwrite(os.path.join(out_dir, f"{base}_bottom_{bw}px.png"), bottom)
    cv2.imwrite(os.path.join(out_dir, f"{base}_left_{bw}px.png"), left)
    cv2.imwrite(os.path.join(out_dir, f"{base}_right_{bw}px.png"), right)

    print("Saved borders to:", os.path.abspath(out_dir))

if __name__ == "__main__":
    image_path = "sliced_images_v1/pinguino_9slices/pinguino_slice_000.png"
    out_dir = "borders_pinguino_slice_000"
    extract_borders(image_path, out_dir, border_width=100)