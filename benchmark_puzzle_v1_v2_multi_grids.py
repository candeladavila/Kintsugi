import argparse
import importlib.util
import os
import random
import time
import signal
import ssl
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import cv2
import numpy as np

# --- SOLUCIÓN ERROR SSL EN MACOS ---
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# Clases de Soporte (Compatibilidad)
# -----------------------------
@dataclass
class SliceV2Adapter:
    id: int
    filename: str
    original_image: np.ndarray
    current_rotation: int = 0
    def set_rotation(self, angle: int) -> None:
        self.current_rotation = int(angle) % 360

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame): raise TimeoutException("Timeout")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

# -----------------------------
# Evaluación: Border Accuracy (Tu lógica de puzzle_base.py)
# -----------------------------
def evaluate_grid(grid, gt_pos):
    """
    Calcula las conexiones correctas comparando vecinos.
    """
    if not grid or not isinstance(grid, list): return {"conexiones_correctas": 0.0}
    
    rows, cols = len(grid), len(grid[0])
    correct_borders = 0
    total_borders = 0
    
    # Mapa inverso: (fila, col) -> PID original
    pos_to_pid = {pos: pid for pid, pos in gt_pos.items()}

    for r in range(rows):
        for c in range(cols):
            piece = grid[r][c]
            pid = getattr(piece, 'id', -1)
            if pid not in gt_pos: continue
            
            gt_r, gt_c = gt_pos[pid]

            # Verificar vecino derecho
            if c < cols - 1:
                total_borders += 1
                right_piece = grid[r][c + 1]
                right_pid = getattr(right_piece, 'id', -1)
                expected_pid = pos_to_pid.get((gt_r, gt_c + 1))
                if expected_pid is not None and right_pid == expected_pid:
                    correct_borders += 1

            # Verificar vecino inferior
            if r < rows - 1:
                total_borders += 1
                bottom_piece = grid[r + 1][c]
                bottom_pid = getattr(bottom_piece, 'id', -1)
                expected_pid = pos_to_pid.get((gt_r + 1, gt_c))
                if expected_pid is not None and bottom_pid == expected_pid:
                    correct_borders += 1

    return {"conexiones_correctas": correct_borders / total_borders if total_borders > 0 else 0.0}

# -----------------------------
# Utilidades de Importación
# -----------------------------
def import_class_from_file(py_path: str, class_name: str) -> Any:
    py_path = os.path.abspath(py_path)
    if not os.path.exists(py_path): 
        print(f"  [!] Archivo no encontrado: {py_path}")
        return None
    try:
        mod_name = f"bench_{os.path.splitext(os.path.basename(py_path))[0]}"
        spec = importlib.util.spec_from_file_location(mod_name, py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name, None)
        if cls is None:
            print(f"  [!] Clase {class_name} no encontrada en {py_path}")
        return cls
    except Exception as e:
        print(f"  [!] Error cargando {py_path}: {e}")
        return None

def try_instantiate_solver(cls, border_width, out_dir):
    """Prueba constructores de PuzzleSolverBase."""
    for ctor in [
        lambda: cls("d", out_dir, "b", border_width), 
        lambda: cls("d", out_dir)
    ]:
        try: return ctor()
        except: continue
    return None

def capture_grid_from_solver(solver):
    captured = {}
    # Sobrescribimos temporalmente los métodos de guardado para capturar la grid
    if hasattr(solver, "save_results"):
        solver.save_results = lambda g, r, c: captured.update({"g": g})
    if hasattr(solver, "save_solution"):
        solver.save_solution = lambda g: captured.update({"g": g})
    
    res = solver.solve()
    return captured.get("g", res)

# -----------------------------
# Bucle de Benchmark
# -----------------------------
def run_benchmark(args):
    from torchvision.datasets import CIFAR10
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Cargando Dataset...")
    ds = CIFAR10(root=args.dataset_root, train=True, download=True)
    indices = random.sample(range(len(ds)), args.num_images)
    
    solvers = {"v1": {}, "v2": {}}
    print("Iniciando detección de solvers en 'puzzle_reconstructor/'...")
    for v in ["v1", "v2"]:
        for m in ["color", "gradient", "paikin"]:
            path = getattr(args, f"{v}_{m}_path")
            cls_name = getattr(args, f"{v}_{m}_class")
            cls = import_class_from_file(path, cls_name)
            if cls: 
                solvers[v][m] = cls
                print(f"  [OK] Solver cargado: {v}/{m}")

    results_data = []

    for idx, img_idx in enumerate(indices):
        pil_img, _ = ds[img_idx]
        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        side = min(h, w)
        img_sq = img_bgr[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]

        for g_size in args.grid_sizes:
            label = f"{g_size}x{g_size}"
            ts = (args.target_size // g_size) * g_size
            canvas = cv2.resize(img_sq, (ts, ts))
            ph = ts // g_size
            
            pieces, gt_pos = [], {}
            for r in range(g_size):
                for c in range(g_size):
                    pid = r * g_size + c
                    pieces.append(canvas[r*ph:(r+1)*ph, c*ph:(c+1)*ph].copy())
                    gt_pos[pid] = (r, c)

            # --- 1. Baseline Random ---
            p_rand = [SimpleNamespace(id=i, image=p) for i, p in enumerate(pieces)]
            random.shuffle(p_rand)
            r_grid = [p_rand[i:i+g_size] for i in range(0, len(p_rand), g_size)]
            results_data.append({**evaluate_grid(r_grid, gt_pos), "version": "v1", "method": "random", "grid": label})

            # --- 2. Ejecución Solvers ---
            for v in ["v1", "v2"]:
                for m_name, cls in solvers[v].items():
                    print(f"Ejecutando: Img {idx+1}/{args.num_images} | {v}/{m_name} | {label}...")
                    solver = try_instantiate_solver(cls, args.border_width, args.output_dir)
                    if not solver: continue
                    
                    solver.slices = [SimpleNamespace(id=i, image=p) for i, p in enumerate(pieces)] if v=="v1" \
                                   else [SliceV2Adapter(i, "p.png", p) for i, p in enumerate(pieces)]
                    
                    try:
                        with time_limit(args.timeout):
                            grid = capture_grid_from_solver(solver)
                        if grid:
                            res = evaluate_grid(grid, gt_pos)
                            results_data.append({**res, "version": v, "method": m_name, "grid": label})
                    except Exception as e:
                        print(f"  [!] Error ejecutando {v}/{m_name}: {e}")
                        continue

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"\nBenchmark finalizado. JSON generado en {args.output_dir}/results.json")

def parse_grid_sizes(s): return [int(x) for x in s.split(",") if x.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--grid-sizes", type=parse_grid_sizes, default="3,5,10")
    parser.add_argument("--target-size", type=int, default=300)
    parser.add_argument("--border-width", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60) # Aumentado por seguridad
    parser.add_argument("--dataset-root", type=str, default="datasets")
    parser.add_argument("--output-dir", type=str, default="benchmark_output")
    
    # V1 solvers from puzzle_reconstructor/
    for m in ["color", "gradient", "paikin"]:
        parser.add_argument(f"--v1-{m}-path", type=str, 
                           default=os.path.join("puzzle_reconstructor", f"{m}_reconstructor.py"))
        parser.add_argument(f"--v1-{m}-class", type=str, 
                           default=f"{m.capitalize()}Solver")
    
    # V2 solvers from puzzle_reconstructor_v2/
    for m in ["color", "gradient", "paikin"]:
        parser.add_argument(f"--v2-{m}-path", type=str, 
                           default=os.path.join("puzzle_reconstructor_v2", f"{m}_reconstructor_v2.py"))
        parser.add_argument(f"--v2-{m}-class", type=str, 
                           default=f"{m.capitalize()}SolverV2")
            
    run_benchmark(parser.parse_args())