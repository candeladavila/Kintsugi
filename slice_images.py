import os
import math
import cv2
import numpy as np
import argparse
import random


def slice_image(image_path, num_slices, output_dir="sliced_images"):
    """
    Divide una imagen en num_slices partes cuadradas y genera un archivo de texto
    con el orden correcto para la recomposición.
    
    Args:
        image_path (str): Ruta a la imagen a dividir
        num_slices (int): Número de partes (debe tener raíz cuadrada exacta)
        output_dir (str): Carpeta donde guardar las partes
    """
    # Verificar que num_slices tiene raíz cuadrada exacta
    sqrt_slices = int(math.sqrt(num_slices))
    if sqrt_slices * sqrt_slices != num_slices:
        raise ValueError(f"El número {num_slices} no tiene raíz cuadrada exacta")
    
    # Crear directorio de salida específico para esta imagen y número de slices
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    specific_output_dir = os.path.join(output_dir, f"{base_name}_{num_slices}slices")
    os.makedirs(specific_output_dir, exist_ok=True)
    
    # Abrir la imagen
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen: {e}")
    
    # Obtener dimensiones de la imagen (OpenCV usa altura, ancho, canales)
    img_height, img_width = img.shape[:2]
    
    # Calcular dimensiones de cada trozo
    slice_width = img_width // sqrt_slices
    slice_height = img_height // sqrt_slices
    
    # Obtener nombre base del archivo sin extensión
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Dividiendo imagen {image_path} en {num_slices} partes ({sqrt_slices}x{sqrt_slices})")
    print(f"Guardando en carpeta: {specific_output_dir}")
    print(f"Dimensiones originales: {img_width}x{img_height}")
    print(f"Dimensiones de cada trozo: {slice_width}x{slice_height}")
    
    # Lista para almacenar el orden de los trozos
    slice_order = []
    
    # Dividir la imagen
    slice_index = 0
    slice_positions = []
    
    # Crear lista de posiciones y mezclarla aleatoriamente
    for row in range(sqrt_slices):
        for col in range(sqrt_slices):
            slice_positions.append((row, col, slice_index))
            slice_index += 1
    
    # Mezclar las posiciones aleatoriamente para guardar en orden aleatorio
    random.shuffle(slice_positions)
    
    # Procesar cada trozo en orden aleatorio
    for idx, (row, col, original_position) in enumerate(slice_positions):
        # Calcular coordenadas del trozo
        left = col * slice_width
        top = row * slice_height
        right = left + slice_width
        bottom = top + slice_height
        
        # Extraer el trozo usando OpenCV (y:y+h, x:x+w)
        slice_img = img[top:bottom, left:right]
        
        # Generar nombre del archivo del trozo (usando índice aleatorio)
        slice_filename = f"{base_name}_slice_{idx:03d}.png"
        slice_path = os.path.join(specific_output_dir, slice_filename)
        
        # Guardar el trozo
        cv2.imwrite(slice_path, slice_img)
        
        # Agregar información del orden para recomposición
        slice_order.append({
            'filename': slice_filename,
            'row': row,
            'col': col,
            'original_position': original_position,
            'saved_as_index': idx,
            'coordinates': {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        })
    
    # Generar archivo de texto con el orden correcto
    order_filename = f"{base_name}_order.txt"
    order_path = os.path.join(specific_output_dir, order_filename)
    
    with open(order_path, 'w', encoding='utf-8') as f:
        f.write(f"Información de recomposición para: {os.path.basename(image_path)}\n")
        f.write(f"Imagen original: {img_width}x{img_height}\n")
        f.write(f"División: {sqrt_slices}x{sqrt_slices} ({num_slices} trozos)\n")
        f.write(f"Tamaño de cada trozo: {slice_width}x{slice_height}\n")
        f.write("NOTA: Los trozos fueron guardados en ORDEN ALEATORIO\n")
        f.write("-" * 50 + "\n\n")
        
        # Escribir información detallada de cada trozo (ordenado por posición original)
        f.write("ORDEN CORRECTO PARA RECOMPOSICIÓN:\n")
        f.write("Pos.Orig | Archivo Guardado    | Fila | Col | Índice Guardado\n")
        f.write("-" * 65 + "\n")
        
        # Ordenar por posición original para mostrar el orden correcto
        sorted_slices = sorted(slice_order, key=lambda x: x['original_position'])
        
        for slice_info in sorted_slices:
            f.write(f"{slice_info['original_position']:8d} | {slice_info['filename']:18s} | "
                   f"{slice_info['row']:4d} | {slice_info['col']:3d} | "
                   f"{slice_info['saved_as_index']:14d}\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write("MAPEO DE ARCHIVOS (Archivo -> Posición Original):\n")
        # Ordenar por índice de archivo guardado
        sorted_by_file = sorted(slice_order, key=lambda x: x['saved_as_index'])
        for slice_info in sorted_by_file:
            f.write(f"{slice_info['filename']} -> Posición original {slice_info['original_position']} "
                   f"(Fila {slice_info['row']}, Col {slice_info['col']})\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write("COORDENADAS ORIGINALES:\n")
        for slice_info in sorted_slices:
            coords = slice_info['coordinates']
            f.write(f"{slice_info['filename']}: "
                   f"({coords['left']}, {coords['top']}) -> "
                   f"({coords['right']}, {coords['bottom']})\n")
    
    print(f"✓ Imagen dividida exitosamente en {num_slices} partes")
    print(f"✓ Trozos guardados en: {specific_output_dir}/")
    print(f"✓ Archivo de orden creado: {order_path}")
    
    return slice_order


def reconstruct_image(order_file_path, output_path="reconstructed_image.png"):
    """
    Recompone una imagen a partir del archivo de orden.
    
    Args:
        order_file_path (str): Ruta al archivo de orden .txt
        output_path (str): Ruta donde guardar la imagen reconstruida
    """
    # Leer el archivo de orden
    with open(order_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraer información de la cabecera
    img_dimensions = None
    division = None
    slice_size = None
    
    for line in lines:
        if line.startswith("Imagen original:"):
            dims = line.split(":")[1].strip().split("x")
            img_dimensions = (int(dims[0]), int(dims[1]))
        elif line.startswith("División:"):
            div_info = line.split(":")[1].strip().split("x")
            division = (int(div_info[0]), int(div_info[1]))
        elif line.startswith("Tamaño de cada trozo:"):
            size_info = line.split(":")[1].strip().split("x")
            slice_size = (int(size_info[0]), int(size_info[1]))
    
    if not all([img_dimensions, division, slice_size]):
        raise ValueError("No se pudo extraer la información necesaria del archivo de orden")
    
    # Crear imagen vacía para la reconstrucción (altura, ancho, canales)
    reconstructed = np.zeros((img_dimensions[1], img_dimensions[0], 3), dtype=np.uint8)
    
    # Directorio donde están los trozos (mismo directorio que el archivo de orden)
    slices_dir = os.path.dirname(order_file_path)
    
    # Leer información de los trozos
    start_reading = False
    for line in lines:
        if line.startswith("Fila | Col"):
            start_reading = True
            continue
        elif start_reading and line.strip() and not line.startswith("-"):
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 4:
                row = int(parts[0])
                col = int(parts[1])
                filename = parts[3]
                
                # Cargar el trozo
                slice_path = os.path.join(slices_dir, filename)
                if os.path.exists(slice_path):
                    slice_img = cv2.imread(slice_path)
                    
                    if slice_img is not None:
                        # Calcular posición en la imagen reconstruida
                        x = col * slice_size[0]
                        y = row * slice_size[1]
                        
                        # Obtener dimensiones del trozo
                        slice_height, slice_width = slice_img.shape[:2]
                        
                        # Pegar el trozo en la posición correcta
                        reconstructed[y:y+slice_height, x:x+slice_width] = slice_img
                    else:
                        print(f"Advertencia: No se pudo cargar la imagen {filename}")
                else:
                    print(f"Advertencia: No se encontró el archivo {filename}")
    
    # Guardar imagen reconstruida
    cv2.imwrite(output_path, reconstructed)
    print(f"✓ Imagen reconstruida guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Dividir y recomponer imágenes en trozos')
    parser.add_argument('action', choices=['slice', 'reconstruct'], 
                       help='Acción a realizar: slice (dividir) o reconstruct (recomponer)')
    parser.add_argument('input_path', help='Ruta de la imagen (para slice) o archivo de orden (para reconstruct)')
    parser.add_argument('-n', '--num-slices', type=int, 
                       help='Número de trozos (solo para slice, debe tener raíz cuadrada exacta)')
    parser.add_argument('-o', '--output', 
                       help='Directorio de salida (para slice) o archivo de imagen (para reconstruct)')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'slice':
            if not args.num_slices:
                print("Error: Se requiere especificar el número de trozos con -n")
                return
            
            output_dir = args.output if args.output else "sliced_images"
            slice_image(args.input_path, args.num_slices, output_dir)
            
        elif args.action == 'reconstruct':
            output_path = args.output if args.output else "reconstructed_image.png"
            reconstruct_image(args.input_path, output_path)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Configuración directa - Modifica estas variables según tus necesidades
    IMAGE_FILE = "images/apple.png"  # Cambia por la ruta de tu imagen
    NUM_SLICES = 4                       # Cambia por el número de trozos (debe tener raíz cuadrada exacta)
    OUTPUT_DIR = "sliced_images"         # Carpeta donde guardar los trozos
    
    import sys
    
    print("=== DIVISOR DE IMÁGENES ===")
    print("Divide imágenes en trozos cuadrados para crear puzzles")
    print("")
    
    # Obtener datos del usuario
    if len(sys.argv) >= 3:
        # Usar argumentos de línea de comandos
        image_path = sys.argv[1]
        try:
            num_slices = int(sys.argv[2])
        except ValueError:
            print("❌ Error: El número de slices debe ser un entero")
            sys.exit(1)
    else:
        # Modo interactivo simplificado
        image_path = input("Ruta de la imagen: ").strip()
        if not image_path:
            image_path = IMAGE_FILE
            
        try:
            num_input = input(f"Número de trozos (Enter para {NUM_SLICES}): ").strip()
            if num_input:
                num_slices = int(num_input)
            else:
                num_slices = NUM_SLICES
        except ValueError:
            print(f"Valor inválido, usando {NUM_SLICES}")
            num_slices = NUM_SLICES
    
    print(f"\n🎯 Configuración:")
    print(f"   Imagen: {image_path}")
    print(f"   Trozos: {num_slices}")
    print(f"   Carpeta de salida: {OUTPUT_DIR}")
    
    try:
        slice_image(image_path, num_slices, OUTPUT_DIR)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Sugerencias:")
        print("   • Verifica que el archivo de imagen existe")
        print("   • Asegúrate de que el número de trozos tenga raíz cuadrada exacta (4, 9, 16, 25, etc.)")
        sys.exit(1)
