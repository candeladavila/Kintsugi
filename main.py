#!/usr/bin/env python3
"""
KINTSUGI - Sistema completo de división y reconstrucción de puzzles de imágenes
Combina la funcionalidad de slice_images.py y puzzle_solver.py en un flujo automático
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

def run_slice_images(image_path, num_slices):
    """
    Ejecuta slice_images.py con los parámetros especificados
    """
    print("🔪 Iniciando división de imagen...")
    print("-" * 40)
    
    try:
        # Ejecutar slice_images.py con argumentos
        cmd = [sys.executable, "slice_images.py", image_path, str(num_slices)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Advertencias:", result.stderr)
        
        print("✅ División completada exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error durante la división: {e}")
        if e.stdout:
            print("Salida:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Error: No se encontró slice_images.py")
        return False

def run_puzzle_solver(image_name, num_slices, method='all'):
    """
    Ejecuta puzzle_solver.py con la configuración especificada
    """
    print(f"\n🧩 Iniciando reconstrucción de puzzle...")
    print("-" * 40)
    
    try:
        # Buscar la carpeta específica creada por slice_images
        sliced_dir = f"sliced_images/{image_name}_{num_slices}slices"
        if not os.path.exists(sliced_dir):
            print(f"❌ Error: No se encontró la carpeta {sliced_dir}")
            return False
        
        # Crear manualmente los solvers con las rutas correctas
        print(f"📂 Usando carpeta de trozos: {sliced_dir}")
        
        # Importar los módulos necesarios
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'puzzle_reconstructor'))
        from gradient_reconstructor import GradientSolver
        from color_reconstructor import ColorSolver
        from random_reconstructor import RandomSolver
        
        output_dir = "output_images"
        
        # Ejecutar métodos según la selección
        if method == 'all':
            methods_to_run = [
                ('gradient', GradientSolver),
                ('color', ColorSolver),
                ('random', RandomSolver)
            ]
        else:
            solver_map = {
                'gradient': GradientSolver,
                'color': ColorSolver,
                'random': RandomSolver
            }
            if method not in solver_map:
                print(f"❌ Error: Método desconocido '{method}'")
                return False
            methods_to_run = [(method, solver_map[method])]
        
        success_count = 0
        for method_name, solver_class in methods_to_run:
            try:
                print(f"\n🔄 Ejecutando método: {method_name.upper()}")
                solver = solver_class(sliced_dir, output_dir, image_name)
                solver.load_slices(image_name)
                solver.solve()
                success_count += 1
            except Exception as e:
                print(f"❌ Error en método {method_name}: {e}")
        
        if success_count > 0:
            print(f"\n✅ Reconstrucción completada: {success_count} métodos exitosos")
            return True
        else:
            print("❌ Todos los métodos fallaron")
            return False
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("Verificar que los módulos puzzle_reconstructor están disponibles")
        return False
    except Exception as e:
        print(f"❌ Error durante la reconstrucción: {e}")
        return False

def get_image_name(image_path):
    """Extrae el nombre base de la imagen sin extensión"""
    return Path(image_path).stem

def validate_image_exists(image_path):
    """Valida que el archivo de imagen exista"""
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encontró el archivo {image_path}")
        return False
    
    # Verificar que sea un archivo de imagen válido
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    ext = Path(image_path).suffix.lower()
    
    if ext not in valid_extensions:
        print(f"❌ Error: {ext} no es un formato de imagen soportado")
        print(f"Formatos válidos: {', '.join(valid_extensions)}")
        return False
    
    return True

def validate_num_slices(num_slices):
    """Valida que el número de slices tenga raíz cuadrada exacta"""
    import math
    sqrt_slices = int(math.sqrt(num_slices))
    if sqrt_slices * sqrt_slices != num_slices:
        print(f"❌ Error: {num_slices} no tiene raíz cuadrada exacta")
        print(f"Números válidos: 4, 9, 16, 25, 36, 49, 64, 81, 100, etc.")
        return False
    return True

def interactive_mode():
    """Modo interactivo para obtener parámetros del usuario"""
    print("🖼️  KINTSUGI - SISTEMA DE PUZZLES DE IMÁGENES")
    print("=" * 50)
    print("Divide una imagen en trozos y luego intenta reconstruirla automáticamente")
    print("")
    
    # Obtener ruta de imagen
    while True:
        image_path = input("📁 Ruta de la imagen: ").strip()
        if not image_path:
            print("❌ Por favor, introduce una ruta válida")
            continue
        
        if validate_image_exists(image_path):
            break
    
    # Obtener número de slices
    while True:
        try:
            num_input = input("🔢 Número de trozos (4, 9, 16, 25, etc.): ").strip()
            num_slices = int(num_input)
            
            if validate_num_slices(num_slices):
                break
        except ValueError:
            print("❌ Por favor, introduce un número entero válido")
    
    # Seleccionar método de reconstrucción
    methods = {
        '1': ('gradient', 'Análisis de gradientes'),
        '2': ('color', 'Análisis de colores'),
        '3': ('random', 'Orden aleatorio'),
        '4': ('all', 'Todos los métodos')
    }
    
    print("\n🔧 Métodos de reconstrucción:")
    for key, (method, desc) in methods.items():
        print(f"  {key}. {method.upper()} - {desc}")
    
    while True:
        choice = input("\n🎯 Selecciona método (1-4, Enter para todos): ").strip()
        
        if not choice:  # Enter presionado
            method = 'all'
            break
        elif choice in methods:
            method = methods[choice][0]
            break
        else:
            print("❌ Por favor, selecciona una opción válida (1-4)")
    
    return image_path, num_slices, method

def main():
    """Función principal"""
    print()
    
    # Verificar que los scripts necesarios existen
    required_files = ['slice_images.py', 'puzzle_solver.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Error: No se encontró {file}")
            print("Asegúrate de ejecutar este script desde el directorio correcto")
            return
    
    # Procesar argumentos de línea de comandos
    if len(sys.argv) >= 3:
        # Modo línea de comandos
        image_path = sys.argv[1]
        try:
            num_slices = int(sys.argv[2])
        except ValueError:
            print("❌ Error: El número de slices debe ser un entero")
            return
        
        method = sys.argv[3] if len(sys.argv) > 3 else 'all'
        
        # Validaciones
        if not validate_image_exists(image_path):
            return
        if not validate_num_slices(num_slices):
            return
        
    else:
        # Modo interactivo
        image_path, num_slices, method = interactive_mode()
    
    # Obtener nombre base de la imagen
    image_name = get_image_name(image_path)
    
    print(f"\n🎯 Configuración:")
    print(f"   Imagen: {image_path}")
    print(f"   Nombre base: {image_name}")
    print(f"   Trozos: {num_slices}")
    print(f"   Método: {method.upper()}")
    print("")
    
    # Confirmar antes de proceder
    if len(sys.argv) < 3:  # Solo en modo interactivo
        confirm = input("¿Continuar? (Enter para sí, 'n' para no): ").strip().lower()
        if confirm == 'n':
            print("🚫 Operación cancelada")
            return
    
    print("\n" + "=" * 60)
    print("🚀 INICIANDO PROCESO COMPLETO")
    print("=" * 60)
    
    # Paso 1: Dividir imagen
    success = run_slice_images(image_path, num_slices)
    if not success:
        print("\n💥 Falló la división de imagen. Proceso terminado.")
        return
    
    # Paso 2: Reconstruir puzzle
    success = run_puzzle_solver(image_name, num_slices, method)
    if not success:
        print("\n💥 Falló la reconstrucción del puzzle.")
        return
    
    # Resumen final
    print("\n" + "=" * 60)
    print("🎉 PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"📁 Trozos guardados en: sliced_images/{image_name}_{num_slices}slices/")
    print(f"🎨 Resultados en: output_images/{image_name}_{num_slices}slices/")
    print("\n✨ ¡Revisa los resultados y compara los diferentes métodos!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)