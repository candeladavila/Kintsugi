#!/usr/bin/env python3
"""
Solucionador de puzzles de imágenes cortadas.
Usa diferentes métodos para reconstruir imágenes divididas en trozos.
"""

import os
import sys
import glob
import argparse

# Agregar la carpeta puzzle_reconstructor al path de Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'puzzle_reconstructor'))

from gradient_reconstructor import GradientSolver
from color_reconstructor import ColorSolver
from random_reconstructor import RandomSolver


def find_available_images(sliced_dir="sliced_images"):
    """
    Encuentra todas las imágenes disponibles en la carpeta de trozos.
    Busca en subcarpetas con formato nombre_Nslices.
    Retorna una lista de tuplas (nombre_base, num_slices, ruta_carpeta)
    """
    if not os.path.exists(sliced_dir):
        return []
    
    available_configs = []
    
    # Buscar subcarpetas con formato nombre_Nslices
    for item in os.listdir(sliced_dir):
        item_path = os.path.join(sliced_dir, item)
        if os.path.isdir(item_path) and "_" in item and item.endswith("slices"):
            try:
                # Extraer nombre base y número de slices
                parts = item.rsplit("_", 1)  # Dividir por el último _
                if len(parts) == 2 and parts[1].endswith("slices"):
                    name_base = parts[0]
                    num_slices = int(parts[1].replace("slices", ""))
                    
                    # Verificar que la carpeta contenga archivos slice
                    slice_files = glob.glob(os.path.join(item_path, "*_slice_*.png"))
                    if slice_files:
                        available_configs.append((name_base, num_slices, item_path))
            except (ValueError, IndexError):
                continue
    
    return sorted(available_configs, key=lambda x: (x[0], x[1]))


def get_slice_count(base_name, slices_path):
    """Cuenta cuántos trozos hay para una configuración específica."""
    pattern = os.path.join(slices_path, f"{base_name}_slice_*.png")
    files = glob.glob(pattern)
    return len(files)


def show_available_images(sliced_dir="sliced_images"):
    """Muestra las imágenes disponibles para reconstruir."""
    configs = find_available_images(sliced_dir)
    
    if not configs:
        print(f"❌ No se encontraron imágenes cortadas en '{sliced_dir}'")
        return []
    
    print(f"\n📁 Configuraciones disponibles en '{sliced_dir}':")
    print("-" * 50)
    
    for i, (name_base, num_slices, slices_path) in enumerate(configs, 1):
        actual_count = get_slice_count(name_base, slices_path)
        print(f"{i:2d}. {name_base} - {num_slices} trozos ({actual_count} archivos)")
    
    return configs


def get_user_choice():
    """Interfaz para que el usuario seleccione imagen y método."""
    
    # Configuración de directorios
    SLICED_DIR = "sliced_images"
    OUTPUT_DIR = "output_images"
    
    print("🧩 SOLUCIONADOR DE PUZZLES DE IMÁGENES")
    print("=" * 50)
    
    # Mostrar imágenes disponibles
    available_configs = show_available_images(SLICED_DIR)
    
    if not available_configs:
        print("\n💡 Primero debes dividir una imagen usando slice_images.py")
        return None, None, None, None
    
    # Selección de configuración
    while True:
        try:
            choice = input(f"\n🎯 Selecciona una configuración (1-{len(available_configs)}): ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                return None, None, None, None
            
            idx = int(choice) - 1
            if 0 <= idx < len(available_configs):
                selected_name, selected_slices, selected_path = available_configs[idx]
                break
            else:
                print(f"❌ Por favor, introduce un número entre 1 y {len(available_configs)}")
        except ValueError:
            print("❌ Por favor, introduce un número válido")
    
    # Mostrar información de la configuración seleccionada
    actual_slice_count = get_slice_count(selected_name, selected_path)
    print(f"\n✅ Configuración seleccionada: {selected_name}")
    print(f"📊 Número de trozos: {selected_slices} ({actual_slice_count} archivos)")
    print(f"📂 Carpeta: {selected_path}")
    
    # Selección de método
    methods = {
        '1': ('gradient', 'Análisis de gradientes (detecta bordes y líneas)'),
        '2': ('color', 'Análisis de colores (continuidad cromática)'),
        '3': ('random', 'Orden aleatorio (sin análisis)'),
        '4': ('all', 'Ejecutar todos los métodos')
    }
    
    print("\n🔧 Métodos de reconstrucción disponibles:")
    print("-" * 50)
    for key, (method_name, description) in methods.items():
        print(f"{key}. {method_name.upper():<12} - {description}")
    
    while True:
        method_choice = input("\n🎯 Selecciona un método (1-4): ").strip()
        
        if method_choice in methods:
            selected_method = methods[method_choice][0]
            break
        else:
            print("❌ Por favor, selecciona una opción válida (1-4)")
    
    return selected_name, selected_method, selected_path, OUTPUT_DIR


def run_solver(image_name, method, slices_path, output_dir):
    """Ejecuta el solucionador especificado."""
    
    print(f"\n🚀 Iniciando reconstrucción con método: {method.upper()}")
    print("-" * 50)
    
    try:
        if method == 'gradient':
            solver = GradientSolver(slices_path, output_dir, image_name)
        elif method == 'color':
            solver = ColorSolver(slices_path, output_dir, image_name)
        elif method == 'random':
            solver = RandomSolver(slices_path, output_dir, image_name)
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        solver.load_slices(image_name)
        solver.solve()
        
        print(f"✅ Reconstrucción completada con método: {method.upper()}")
        
    except Exception as e:
        print(f"❌ Error durante la reconstrucción: {e}")
        return False
    
    return True


def main():
    """Función principal del programa."""
    
    # Manejo de argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Solucionador de puzzles de imágenes cortadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python puzzle_solver.py                    # Modo interactivo
  python puzzle_solver.py -i imagen -m gradient
  python puzzle_solver.py -i imagen -m all
        """
    )
    
    parser.add_argument('-i', '--image', 
                       help='Nombre de la imagen base (sin _slice_XXX.png)')
    parser.add_argument('-m', '--method', 
                       choices=['gradient', 'color', 'random', 'all'],
                       help='Método de reconstrucción')
    parser.add_argument('--sliced-dir', default='sliced_images',
                       help='Directorio con los trozos (default: sliced_images)')
    parser.add_argument('--output-dir', default='output_images',
                       help='Directorio de salida (default: output_images)')
    
    args = parser.parse_args()
    
    # Modo línea de comandos
    if args.image and args.method:
        image_name = args.image
        method = args.method
        sliced_dir = args.sliced_dir
        output_dir = args.output_dir
        
        # Verificar que la imagen existe
        available_images = find_available_images(sliced_dir)
        if image_name not in available_images:
            print(f"❌ Error: No se encontró la imagen '{image_name}' en '{sliced_dir}'")
            print(f"📋 Imágenes disponibles: {', '.join(available_images)}")
            return
        
    # Modo interactivo
    else:
        image_name, method, sliced_dir, output_dir = get_user_choice()
        
        if not all([image_name, method, sliced_dir, output_dir]):
            print("\n👋 ¡Hasta luego!")
            return
    
    # Ejecutar reconstrucción
    if method == 'all':
        methods_to_run = ['gradient', 'color', 'random']
        
        print(f"\n🔄 Ejecutando todos los métodos para: {image_name}")
        print("=" * 60)
        
        results = []
        for single_method in methods_to_run:
            success = run_solver(image_name, single_method, sliced_dir, output_dir)
            results.append((single_method, success))
        
        # Resumen final
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE RESULTADOS")
        print("=" * 60)
        
        successful = [method for method, success in results if success]
        failed = [method for method, success in results if not success]
        
        if successful:
            print(f"✅ Métodos exitosos: {', '.join(successful).upper()}")
        
        if failed:
            print(f"❌ Métodos fallidos: {', '.join(failed).upper()}")
        
        if successful:
            print(f"\n📁 Resultados guardados en: {output_dir}/")
            print("🎨 Compara los diferentes métodos para ver cuál funciona mejor!")
    
    else:
        success = run_solver(image_name, method, sliced_dir, output_dir)
        
        if success:
            print(f"\n📁 Resultado guardado en: {output_dir}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)