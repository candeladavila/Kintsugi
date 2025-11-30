import os
import math
import cv2
import numpy as np
import random
import glob


def organize_slices_randomly(original_image_name, num_slices, sliced_dir="sliced_images", output_dir="output_images"):
    """
    Reorganiza los trozos de una imagen de forma aleatoria y genera una nueva imagen.
    
    Args:
        original_image_name (str): Nombre de la imagen original (sin extensión)
        num_slices (int): Número de trozos en que se dividió la imagen
        sliced_dir (str): Carpeta donde están los trozos
        output_dir (str): Carpeta donde guardar la imagen reorganizada
    """
    # Verificar que num_slices tiene raíz cuadrada exacta
    sqrt_slices = int(math.sqrt(num_slices))
    if sqrt_slices * sqrt_slices != num_slices:
        raise ValueError(f"El número {num_slices} no tiene raíz cuadrada exacta")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar todos los trozos de la imagen
    slice_pattern = os.path.join(sliced_dir, f"{original_image_name}_slice_*.png")
    slice_files = glob.glob(slice_pattern)
    
    if len(slice_files) != num_slices:
        raise ValueError(f"Se esperaban {num_slices} trozos, pero se encontraron {len(slice_files)}")
    
    # Ordenar los archivos por número de slice para tener el orden original
    slice_files.sort(key=lambda x: int(x.split('_slice_')[1].split('.')[0]))
    
    print(f"Encontrados {len(slice_files)} trozos de la imagen '{original_image_name}'")
    
    # Cargar el primer trozo para obtener las dimensiones
    first_slice = cv2.imread(slice_files[0])
    if first_slice is None:
        raise ValueError(f"No se pudo cargar el primer trozo: {slice_files[0]}")
    
    slice_height, slice_width = first_slice.shape[:2]
    
    # Calcular dimensiones de la imagen completa
    total_width = slice_width * sqrt_slices
    total_height = slice_height * sqrt_slices
    
    print(f"Dimensiones de cada trozo: {slice_width}x{slice_height}")
    print(f"Dimensiones de la imagen final: {total_width}x{total_height}")
    
    # Crear una lista con los índices de posición original
    original_positions = list(range(num_slices))
    
    # Crear una lista con las posiciones donde colocar cada trozo (mezclada aleatoriamente)
    random_positions = list(range(num_slices))
    random.shuffle(random_positions)
    
    print("Mapeo de reorganización:")
    print("Posición original -> Nueva posición")
    for i, new_pos in enumerate(random_positions):
        orig_row, orig_col = divmod(i, sqrt_slices)
        new_row, new_col = divmod(new_pos, sqrt_slices)
        print(f"({orig_row},{orig_col}) -> ({new_row},{new_col})")
    
    # Crear imagen vacía para la reorganización
    reorganized_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Colocar cada trozo en su nueva posición aleatoria
    for original_idx, new_position in enumerate(random_positions):
        # Cargar el trozo original
        slice_img = cv2.imread(slice_files[original_idx])
        if slice_img is None:
            print(f"Advertencia: No se pudo cargar {slice_files[original_idx]}")
            continue
        
        # Calcular la nueva posición en la imagen reorganizada
        new_row = new_position // sqrt_slices
        new_col = new_position % sqrt_slices
        
        new_y = new_row * slice_height
        new_x = new_col * slice_width
        
        # Colocar el trozo en la nueva posición
        reorganized_img[new_y:new_y+slice_height, new_x:new_x+slice_width] = slice_img
    
    # Generar nombre del archivo de salida
    output_filename = f"{original_image_name}_randomized.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Guardar la imagen reorganizada
    cv2.imwrite(output_path, reorganized_img)
    
    # Generar archivo de texto con el mapeo de reorganización
    mapping_filename = f"{original_image_name}_randomization_map.txt"
    mapping_path = os.path.join(output_dir, mapping_filename)
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.write(f"Mapeo de reorganización aleatoria para: {original_image_name}\n")
        f.write(f"Número de trozos: {num_slices} ({sqrt_slices}x{sqrt_slices})\n")
        f.write(f"Dimensiones de cada trozo: {slice_width}x{slice_height}\n")
        f.write(f"Imagen reorganizada guardada en: {output_filename}\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("MAPEO DE REORGANIZACIÓN:\n")
        f.write("Pos.Original | Nueva Pos. | Fila Orig | Col Orig | Nueva Fila | Nueva Col\n")
        f.write("-" * 70 + "\n")
        
        for original_idx, new_position in enumerate(random_positions):
            orig_row, orig_col = divmod(original_idx, sqrt_slices)
            new_row, new_col = divmod(new_position, sqrt_slices)
            
            f.write(f"{original_idx:12d} | {new_position:10d} | "
                   f"{orig_row:9d} | {orig_col:8d} | "
                   f"{new_row:10d} | {new_col:9d}\n")
        
        f.write(f"\n" + "-" * 50 + "\n")
        f.write("LISTA DE ARCHIVOS UTILIZADOS:\n")
        for i, file_path in enumerate(slice_files):
            filename = os.path.basename(file_path)
            f.write(f"{i:3d}: {filename}\n")
    
    print(f"✓ Imagen reorganizada guardada en: {output_path}")
    print(f"✓ Mapeo de reorganización guardado en: {mapping_path}")
    
    return output_path, mapping_path


def organize_slices_custom(original_image_name, num_slices, custom_order, sliced_dir="sliced_images", output_dir="output_images"):
    """
    Reorganiza los trozos según un orden personalizado.
    
    Args:
        original_image_name (str): Nombre de la imagen original (sin extensión)
        num_slices (int): Número de trozos
        custom_order (list): Lista con el nuevo orden de los trozos
        sliced_dir (str): Carpeta donde están los trozos
        output_dir (str): Carpeta donde guardar la imagen reorganizada
    """
    if len(custom_order) != num_slices:
        raise ValueError(f"El orden personalizado debe tener {num_slices} elementos")
    
    if set(custom_order) != set(range(num_slices)):
        raise ValueError("El orden personalizado debe contener todos los números de 0 a n-1")
    
    sqrt_slices = int(math.sqrt(num_slices))
    if sqrt_slices * sqrt_slices != num_slices:
        raise ValueError(f"El número {num_slices} no tiene raíz cuadrada exacta")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar todos los trozos
    slice_pattern = os.path.join(sliced_dir, f"{original_image_name}_slice_*.png")
    slice_files = glob.glob(slice_pattern)
    slice_files.sort(key=lambda x: int(x.split('_slice_')[1].split('.')[0]))
    
    if len(slice_files) != num_slices:
        raise ValueError(f"Se esperaban {num_slices} trozos, pero se encontraron {len(slice_files)}")
    
    # Cargar dimensiones
    first_slice = cv2.imread(slice_files[0])
    slice_height, slice_width = first_slice.shape[:2]
    total_width = slice_width * sqrt_slices
    total_height = slice_height * sqrt_slices
    
    # Crear imagen vacía
    reorganized_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Colocar trozos según el orden personalizado
    for position, original_idx in enumerate(custom_order):
        slice_img = cv2.imread(slice_files[original_idx])
        
        row = position // sqrt_slices
        col = position % sqrt_slices
        
        y = row * slice_height
        x = col * slice_width
        
        reorganized_img[y:y+slice_height, x:x+slice_width] = slice_img
    
    # Guardar imagen
    output_filename = f"{original_image_name}_custom_order.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, reorganized_img)
    
    print(f"✓ Imagen con orden personalizado guardada en: {output_path}")
    return output_path


if __name__ == "__main__":
    # Configuración directa - Modifica estas variables según tus necesidades
    ORIGINAL_IMAGE_NAME = "apple"  # Nombre de la imagen original (sin extensión)
    NUM_SLICES = 4                  # Número de trozos en que se dividió
    SLICED_DIR = "sliced_images"    # Carpeta donde están los trozos
    OUTPUT_DIR = "output_images"    # Carpeta donde guardar las imágenes reorganizadas
    
    import sys
    
    if len(sys.argv) == 1:
        print("=== REORGANIZADOR DE TROZOS DE IMAGEN ===")
        print(f"Imagen configurada: {ORIGINAL_IMAGE_NAME}")
        print(f"Número de trozos: {NUM_SLICES}")
        print(f"Carpeta de trozos: {SLICED_DIR}")
        print(f"Carpeta de salida: {OUTPUT_DIR}")
        print("")
        
        usar_config = input("¿Usar la configuración predefinida? (s/n): ").strip().lower()
        
        if usar_config in ['s', 'si', 'sí', 'y', 'yes', '']:
            print("Usando configuración predefinida...")
            try:
                organize_slices_randomly(ORIGINAL_IMAGE_NAME, NUM_SLICES, SLICED_DIR, OUTPUT_DIR)
            except Exception as e:
                print(f"Error: {e}")
                print("\nVerifica que:")
                print("1. Los trozos existan en la carpeta especificada")
                print("2. El nombre de la imagen sea correcto")
                print("3. El número de trozos coincida con los archivos encontrados")
        else:
            print("\n=== MODO INTERACTIVO ===")
            image_name = input("Nombre de la imagen original (sin extensión): ").strip()
            num_slices = int(input("Número de trozos: "))
            sliced_dir = input(f"Carpeta de trozos (Enter para '{SLICED_DIR}'): ").strip()
            output_dir = input(f"Carpeta de salida (Enter para '{OUTPUT_DIR}'): ").strip()
            
            if not sliced_dir:
                sliced_dir = SLICED_DIR
            if not output_dir:
                output_dir = OUTPUT_DIR
            
            mode = input("¿Reorganización aleatoria (r) o orden personalizado (p)? ").strip().lower()
            
            try:
                if mode in ['r', 'random', 'aleatoria', 'aleatorio']:
                    organize_slices_randomly(image_name, num_slices, sliced_dir, output_dir)
                elif mode in ['p', 'personal', 'personalizado', 'custom']:
                    print(f"Introduce el orden personalizado (números de 0 a {num_slices-1}):")
                    order_str = input("Separados por espacios o comas: ")
                    custom_order = [int(x.strip()) for x in order_str.replace(',', ' ').split()]
                    organize_slices_custom(image_name, num_slices, custom_order, sliced_dir, output_dir)
                else:
                    print("Opción no válida. Usando reorganización aleatoria...")
                    organize_slices_randomly(image_name, num_slices, sliced_dir, output_dir)
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("Uso: python organize.py")
        print("El script se ejecuta en modo interactivo.")
