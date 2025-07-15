# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import (
    filedialog, Button, Label, Frame, Scale, IntVar, messagebox, simpledialog,
    Text, Scrollbar, Toplevel, Entry
)
import rasterio
from rasterio.plot import show, plotting_extent
from rasterio.warp import calculate_default_transform, reproject, Resampling
# Import shapes function for perimeter calculation
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon, Polygon # To calculate perimeter from shapes
import matplotlib.pyplot as plt
import numpy as np
from pysheds.grid import Grid
import warnings
import os
# Importar para operaciones morfológicas (borde) y contornos
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_erosion, binary_dilation # Importar binary_dilation
# Importar skimage para mejorar la red de drenaje con esqueletonización
try:
    from skimage.morphology import skeletonize, thin
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("skimage no está disponible. Se usarán alternativas para mejorar la red de drenaje.")

# Import math for sqrt and pi
import math
# Import re for string searching in indices extraction
import re
# Import traceback for detailed error printing
import traceback
# Import pysheds para chequear versión (opcional)
import pysheds

# Imprimir versión de PySheds
print(f"PySheds version: {pysheds.__version__}")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pysheds")
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress Shapely warning about projection
warnings.filterwarnings("ignore", category=UserWarning, module="shapely")


# Global variable to store clicked coordinates
clicked_coords = [None, None]
Font_tuple2=("Comic Sans MS", 12, "bold")
# Global variables to store results for morphometry calculation
last_watershed_data = {
    "watershed_mask": None,
    "rivers_mask": None, # Rivers inside watershed (enhanced for display and snap)
    "rivers_for_strahler_mask": None, # NEW: Rivers inside watershed (original, for Strahler calculation/display)
    "dem_data": None,    # DEM values (original read by pysheds)
    "inflated_dem": None,# DEM filled and resolved
    "transform": None,
    "src_res": None,     # (x_res, y_res)
    "is_geographic": None,
    "pysheds_grid": None, # RENOMBRADO: Objeto Grid de PySheds
    "outlet_coords": None, # (row, col)
    "fdir": None,        # Flow direction numpy array
    "strahler_order_data": None, # NUEVO: Datos del orden de Strahler
    "calculated": False
}
morphometry_button = None # To enable/disable the button
runoff_button = None # To enable/disable the button for runoff calculation
hydrograph_button = None # To enable/disable the button for SCS hydrograph
SIMULADOR=None
max_strahler_order = None # Global variable for Strahler interpretation

# --- Funciones de Selección de Archivos ---

def select_dem_file():
    """Abre un diálogo para seleccionar un archivo DEM."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Seleccione el archivo DEM (GeoTIFF)",
        filetypes=[("GeoTIFF files", "*.tif *.tiff")]
    )
    root.destroy()
    if not file_path:
        print("No se seleccionó ningún archivo DEM.")
        return None
    return file_path

def select_watershed_file():
    """Abre un diálogo para seleccionar un archivo de cuenca."""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Seleccione el archivo de CUENCA (GeoTIFF)",
        filetypes=[("GeoTIFF files", "*.tif *.tiff")]
    )
    root.destroy()
    if not file_path:
        print("No se seleccionó ningún archivo de cuenca.")
        return None
    return file_path
# --- Nueva función para mejorar la red de drenaje ---
def enhance_drainage_network(rivers_raster_np, iterations=2):
    """Mejora la continuidad de la red de drenaje usando morfología matemática.
    
    Parámetros:
    rivers_raster_np: Array booleano de la red de drenaje (True donde hay río)
    iterations: Número de iteraciones para las operaciones morfológicas
    
    Retorna:
    Array booleano mejorado de la red de drenaje
    """
    # Verificar si tenemos una red para procesar
    if not np.any(rivers_raster_np):
        return rivers_raster_np
    
    # Guardar la red original para combinar al final
    original_rivers = rivers_raster_np.copy()
    
    # 1. Dilatar para conectar segmentos cercanos
    dilated = binary_dilation(rivers_raster_np, iterations=iterations)
    
    # 2. Esqueletonizar para adelgazar a líneas de 1 píxel
    if HAS_SKIMAGE:
        skeleton = skeletonize(dilated)
        # 3. Combinar la red original con el esqueleto
        enhanced_rivers = np.logical_or(original_rivers, skeleton)
        # 4. Aplicar una dilatación ligera final para mejor visualización
        final_rivers = binary_dilation(enhanced_rivers, iterations=1)
    else:
        # Si no está disponible skimage, usar erosión para adelgazar
        eroded = binary_erosion(dilated, iterations=iterations-1)
        # Ligera dilatación final para conectividad
        final_rivers = binary_dilation(eroded, iterations=1)
    
    return final_rivers

# --- Funciones de Procesamiento Geoespacial ---

def snap_to_river(facc_np, rivers_raster_np, point_coords, transform, search_radius=30):
    """Ajusta las coordenadas del usuario a la celda de río más cercana con mayor acumulación."""
    try:
        row, col = rasterio.transform.rowcol(transform, point_coords[0], point_coords[1])

        if not (0 <= row < facc_np.shape[0] and 0 <= col < facc_np.shape[1]):
             print(f"Error: El punto inicial ({point_coords[0]:.4f}, {point_coords[1]:.4f}) está fuera de los límites del DEM.")
             messagebox.showerror("Error de Coordenadas", f"El punto ({point_coords[0]:.4f}, {point_coords[1]:.4f}) está fuera de los límites del DEM.")
             return None, None

        # search_radius = 30 # Esta línea es redundante, ya que search_radius se pasa como argumento.
        
        min_row = max(0, int(row) - search_radius)
        max_row = min(facc_np.shape[0], int(row) + search_radius + 1)
        min_col = max(0, int(col) - search_radius)
        max_col = min(facc_np.shape[1], int(col) + search_radius + 1)

        facc_sub = facc_np[min_row:max_row, min_col:max_col]
        # CORRECCIÓN: Asegurarse de que la subregión de ríos tenga las mismas dimensiones que la de acumulación
        rivers_sub = rivers_raster_np[min_row:max_row, min_col:max_col] 
        river_indices = np.where(rivers_sub)

        snapped_row, snapped_col = None, None

        if river_indices[0].size > 0:
            river_acc_values = facc_sub[river_indices]
            valid_acc_mask = ~np.isnan(river_acc_values)
            if np.any(valid_acc_mask):
                max_acc_idx = np.argmax(river_acc_values[valid_acc_mask])
                valid_indices = np.where(valid_acc_mask)[0]
                best_idx = valid_indices[max_acc_idx]
                
                snapped_row = min_row + river_indices[0][best_idx]
                snapped_col = min_col + river_indices[1][best_idx]
                print(f"Seleccionando punto con máxima acumulación: {facc_np[snapped_row, snapped_col]}")
            else:
                print(f"Advertencia: Celdas de río encontradas cerca de ({point_coords[0]:.4f}, {point_coords[1]:.4f}), pero con acumulación NaN.")
                if river_indices[0].size > 0:
                    # Fallback a la celda de río más cercana si todos los valores de acumulación son NaN
                    center_row_sub = int(row) - min_row
                    center_col_sub = int(col) - min_col
                    distances = np.sqrt((river_indices[0] - center_row_sub)**2 + (river_indices[1] - center_col_sub)**2)
                    closest_idx = np.argmin(distances)
                    snapped_row = min_row + river_indices[0][closest_idx]
                    snapped_col = min_col + river_indices[1][closest_idx]
                    print("Usando la celda de río más cercana geométricamente debido a acumulación NaN.")

        if snapped_row is None:
            print(f"Advertencia: No se encontraron ríos válidos cerca. Usando punto con mayor acumulación en la ventana de búsqueda.")
            valid_facc_sub_mask = ~np.isnan(facc_sub)
            if np.any(valid_facc_sub_mask):
                 max_acc_val = np.nanmax(facc_sub)
                 max_indices = np.where(facc_sub == max_acc_val)
                 if len(max_indices[0]) > 1:
                     center_row_sub = int(row) - min_row
                     center_col_sub = int(col) - min_col
                     distances = np.sqrt((max_indices[0] - center_row_sub)**2 + (max_indices[1] - center_col_sub)**2)
                     closest_max_idx_in_max_indices = np.argmin(distances)
                     max_acc_idx_local = (max_indices[0][closest_max_idx_in_max_indices], max_indices[1][closest_max_idx_in_max_indices])
                 else:
                     max_acc_idx_local = (max_indices[0][0], max_indices[1][0]) # Take the first one if multiple max

                 snapped_row = min_row + max_acc_idx_local[0]
                 snapped_col = min_col + max_acc_idx_local[1]
            else:
                 print(f"Error: No se encontraron celdas válidas (no-NaN) en la ventana de búsqueda.")
                 messagebox.showerror("Error de Ajuste", "No se encontraron celdas válidas (no-NaN) en la ventana de búsqueda alrededor del punto.")
                 return None, None

        if snapped_row is not None and snapped_col is not None:
            snapped_x, snapped_y = rasterio.transform.xy(transform, snapped_row, snapped_col)
            print(f"Punto ajustado a: ({snapped_x:.4f}, {snapped_y:.4f}) [Fila: {int(snapped_row)}, Col: {int(snapped_col)}]")
            return snapped_row, snapped_col
        else:
             print("Error inesperado al determinar el punto ajustado.")
             messagebox.showerror("Error Interno", "Error inesperado al determinar el punto ajustado.")
             return None, None

    except Exception as e:
        messagebox.showerror("Error en snap_to_river", f"Error: {e}")
        print(f"Error en snap_to_river: {e}")
        traceback.print_exc()
        return None, None

def delineate_watershed_robust(fdir_np, outlet_row, outlet_col, max_cells=None):
    """Versión más robusta de delineación de cuenca que maneja DEM grandes."""
    nrows, ncols = fdir_np.shape
    watershed = np.zeros((nrows, ncols), dtype=bool)

    if not (0 <= outlet_row < nrows and 0 <= outlet_col < ncols):
        print(f"Error: Punto de salida ({outlet_row}, {outlet_col}) fuera de los límites.")
        return watershed  # Devuelve máscara vacía

    # Verificar si el punto de salida tiene una dirección de flujo válida
    outlet_fdir = fdir_np[outlet_row, outlet_col]
    if np.isnan(outlet_fdir) or outlet_fdir <= 0:
         print(f"Error: El punto de salida no tiene dirección de flujo válida (valor={outlet_fdir}).")
         # Intentar buscar un vecino válido en ventana 3x3
         for i in range(-1, 2):
             for j in range(-1, 2):
                 new_row, new_col = outlet_row + i, outlet_col + j
                 if 0 <= new_row < nrows and 0 <= new_col < ncols:
                     neighbor_fdir = fdir_np[new_row, new_col]
                     if not np.isnan(neighbor_fdir) and neighbor_fdir > 0:
                         print(f"Usando celda vecina en ({new_row}, {new_col}) con dirección válida ({neighbor_fdir}).")
                         outlet_row, outlet_col = new_row, new_col
                         outlet_fdir = neighbor_fdir
                         break
             if not np.isnan(outlet_fdir) and outlet_fdir > 0:
                 break

    # Si aún no tenemos dirección válida, devolver vacío
    if np.isnan(outlet_fdir) or outlet_fdir <= 0:
        print("No se pudo encontrar un punto de salida con dirección válida.")
        return watershed

    # Establecer límite del número de celdas si no se proporciona
    if max_cells is None:
        max_cells = min(10000000, nrows * ncols)  # Limitar para evitar bucles infinitos

    # Marcar el punto de salida como parte de la cuenca
    watershed[outlet_row, outlet_col] = True

    # Dirmap estándar D8 (PySheds)
    dir_map = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }

    # Crear inflow_map para seguimiento más eficiente
    print("Creando mapa de dirección de flujo inversa...")
    inflow_map = {}

    # Usar un enfoque por bloques para reducir el uso de memoria
    block_size = 1000  # Procesar el DEM en bloques
    for start_row in range(0, nrows, block_size):
        end_row = min(start_row + block_size, nrows)
        for start_col in range(0, ncols, block_size):
            end_col = min(start_col + block_size, ncols)

            # Procesar este bloque
            block = fdir_np[start_row:end_row, start_col:end_col]
            rows, cols = np.where((~np.isnan(block)) & (block > 0))

            for i in range(len(rows)):
                r, c = rows[i] + start_row, cols[i] + start_col
                direction = fdir_np[r, c]
                dr, dc = dir_map.get(direction, (0, 0))
                if dr == 0 and dc == 0:
                    continue

                target_r, target_c = r + dr, c + dc
                if 0 <= target_r < nrows and 0 <= target_c < ncols:
                    if (target_r, target_c) not in inflow_map:
                        inflow_map[(target_r, target_c)] = []
                    inflow_map[(target_r, target_c)].append((r, c))

    print("Delineando la cuenca (rastreo inverso más eficiente)...")
    queue = [(outlet_row, outlet_col)]
    processed_count = 0

    while queue and processed_count < max_cells:
        current_r, current_c = queue.pop(0)
        processed_count += 1

        if processed_count % 10000 == 0:
            print(f"Procesadas {processed_count} celdas...")

        # Encontrar celdas que fluyen HACIA la celda actual
        incoming_cells = inflow_map.get((current_r, current_c), [])
        for r_in, c_in in incoming_cells:
            if not watershed[r_in, c_in]:
                watershed[r_in, c_in] = True
                queue.append((r_in, c_in))

    cells_in_watershed = np.sum(watershed)
    print(f"Cuenca delineada de manera robusta. Celdas: {cells_in_watershed}")

    return watershed
def find_main_channel_point(facc_np, fdir_np, rivers_raster_np, point_coords, transform, search_radius=100):
    """Encuentra un punto en el cauce principal cercano a las coordenadas dadas"""
    try:
        # Convertir coordenadas a fila/columna
        row, col = rasterio.transform.rowcol(transform, point_coords[0], point_coords[1])
        
        # Definir área de búsqueda amplia
        min_row = max(0, row - search_radius)
        max_row = min(facc_np.shape[0], row + search_radius)
        min_col = max(0, col - search_radius)
        max_col = min(facc_np.shape[1], col + search_radius)
        
        # Extraer subregión de acumulación y ríos
        facc_region = facc_np[min_row:max_row, min_col:max_col]
        rivers_region = rivers_raster_np[min_row:max_row, min_col:max_col]
        
        # Encontrar los 5 puntos con mayor acumulación en la red de drenaje
        river_mask = rivers_region & ~np.isnan(facc_region)
        if not np.any(river_mask):
            print("No se encontraron celdas de río en el área de búsqueda")
            return None, None
        
        # Obtener valores de acumulación para celdas de río
        river_rows, river_cols = np.where(river_mask)
        if len(river_rows) == 0:
            return None, None
            
        river_accs = facc_region[river_mask]
        
        # Ordenar por acumulación (descendente)
        sorted_indices = np.argsort(-river_accs)
        
        # Tomar los puntos con más acumulación (hasta 5)
        num_points = min(5, len(sorted_indices))
        best_candidates = []
        
        for i in range(num_points):
            idx = sorted_indices[i]
            r, c = river_rows[idx], river_cols[idx]
            
            # Convertir a coordenadas globales
            global_r = min_row + r
            global_c = min_col + c
            
            # Verificar si este punto genera una cuenca de tamaño razonable
            test_watershed = delineate_watershed_robust(fdir_np, global_r, global_c, max_cells=10000)
            watershed_size = np.sum(test_watershed)
            
            if watershed_size > 500:  # Al menos 500 celdas
                # Convertir a coordenadas geográficas
                x, y = rasterio.transform.xy(transform, global_r, global_c)
                print(f"Encontrado punto viable en cauce principal: ({x:.4f}, {y:.4f}), tamaño cuenca: {watershed_size}")
                return global_r, global_c
            
            best_candidates.append((global_r, global_c, watershed_size))
        
        # Si ningún punto cumple el criterio de tamaño, usar el que genera la cuenca más grande
        if best_candidates:
            best_candidates.sort(key=lambda x: x[2], reverse=True)
            best_r, best_c, size = best_candidates[0]
            x, y = rasterio.transform.xy(transform, best_r, best_c)
            print(f"Usando mejor candidato disponible: ({x:.4f}, {y:.4f}), tamaño cuenca: {size}")
            return best_r, best_c
        
        return None, None
        
    except Exception as e:
        print(f"Error buscando punto en cauce principal: {e}")
        traceback.print_exc()
        return None, None

def get_cell_area_m2(transform, is_geographic, mean_lat=None):
    """Calcula el área de una celda individual en metros cuadrados."""
    res_x = abs(transform.a) # Resolución en X (ancho de celda en unidades del CRS)
    res_y = abs(transform.e) # Resolución en Y (alto de celda en unidades del CRS)

    if is_geographic:
        # Si es geográfico, res_x y res_y están en GRADOS decimales.
        if mean_lat is None:
            mean_lat = 0 # Aproximación muy burda si no se proporciona latitud
            print("Advertencia: Calculando área de celda geográfica sin latitud media específica. Usando latitud 0.")
        # Factores de conversión aproximados de grados a metros
        m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_lat)) + 1.175 * math.cos(4 * math.radians(mean_lat)) # Más preciso
        m_per_deg_lon = 111412.84 * math.cos(math.radians(mean_lat)) - 93.5 * math.cos(3 * math.radians(mean_lat)) + 0.118 * math.cos(5 * math.radians(mean_lat)) # Más preciso
        # Área de la celda en metros cuadrados
        cell_area_m2 = (res_x * m_per_deg_lon) * (res_y * m_per_deg_lat)
    else:
        # Si es proyectado, se ASUME que las unidades son METROS.
        # ¡¡¡ IMPORTANTE: Si el CRS proyectado usa pies u otras unidades, esto será INCORRECTO !!!
        cell_area_m2 = res_x * res_y

    return cell_area_m2

def calculate_area_km2(watershed_np, transform, is_geographic, src_res=None):
    """Calcula el área total de la cuenca en km²."""
    num_cells = np.sum(watershed_np)
    if num_cells == 0: return 0.0

    # Calcular latitud media si es geográfica para el área de celda
    mean_lat = None
    if is_geographic:
        where_watershed = np.where(watershed_np)
        if where_watershed[0].size > 0:
            # Calcular la latitud media de las celdas de la cuenca
            rows_in_watershed = where_watershed[0]
            cols_in_watershed = where_watershed[1]
            # Obtener coordenadas Y (latitudes) de las celdas
            _, lats = rasterio.transform.xy(transform, rows_in_watershed, cols_in_watershed)
            mean_lat = np.mean(lats)
        else:
            mean_lat = 0 # Fallback si no hay celdas

    # Calcular área de una celda promedio en m²
    cell_area_m2 = get_cell_area_m2(transform, is_geographic, mean_lat)
    # Área total en m²
    total_area_m2 = num_cells * cell_area_m2
    # Convertir a km²
    total_area_km2 = total_area_m2 / 1_000_000.0

    print(f"Área de la cuenca: {total_area_km2:.3f} km² ({num_cells} celdas, área celda aprox: {cell_area_m2:.2f} m²)")
    return total_area_km2

def calculate_perimeter_km(watershed_mask, transform, is_geographic):
    """Calcula el perímetro de la cuenca en km usando rasterio.features.shapes."""
    if not np.any(watershed_mask):
        return 0.0

    mask_uint8 = watershed_mask.astype(rasterio.uint8)
    mask_uint8 = np.ascontiguousarray(mask_uint8)
    perimeter_km = np.nan # Valor por defecto

    try:
        # Extraer formas (polígonos)
        shape_gen = shapes(mask_uint8, mask=watershed_mask, transform=transform)
        basin_shape = None
        total_perimeter = 0.0
        # Sumar la longitud de todos los polígonos extraídos (maneja islas/huecos)
        for geom, value in shape_gen:
            if value == 1: # Valor de las celdas de la cuenca
                current_shape = shape(geom)
                total_perimeter += current_shape.length # Longitud en unidades del CRS
                if basin_shape is None: # Guardar la primera forma para el centroide
                    basin_shape = current_shape

        if basin_shape is None:
            print("Advertencia: No se pudo extraer la forma de la cuenca para el perímetro.")
            return np.nan

        perimeter_native_units = total_perimeter

        if is_geographic:
            # La longitud está en GRADOS. Necesita conversión APROXIMADA a km.
            mean_lat = None
            if basin_shape and basin_shape.centroid:
                mean_lat = basin_shape.centroid.y
            else:
                # Fallback: calcular latitud media de la máscara
                where_watershed = np.where(watershed_mask)
                if where_watershed[0].size > 0:
                    rows_in_watershed = where_watershed[0]
                    cols_in_watershed = where_watershed[1]
                    _, lats = rasterio.transform.xy(transform, rows_in_watershed, cols_in_watershed)
                    mean_lat = np.mean(lats)

            if mean_lat is not None:
                # Factores de conversión aproximados (usando los más precisos)
                m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_lat)) + 1.175 * math.cos(4 * math.radians(mean_lat))
                m_per_deg_lon = 111412.84 * math.cos(math.radians(mean_lat)) - 93.5 * math.cos(3 * math.radians(mean_lat)) + 0.118 * math.cos(5 * math.radians(mean_lat))
                # Usar un promedio simple de los factores (muy aproximado para un perímetro irregular)
                avg_m_per_degree = (m_per_deg_lat + m_per_deg_lon) / 2.0
                perimeter_m = perimeter_native_units * avg_m_per_degree
                perimeter_km = perimeter_m / 1000.0
                print(f"Perímetro (aprox. geográfico): {perimeter_km:.3f} km (longitud nativa: {perimeter_native_units:.4f} grados)")
            else:
                print("Advertencia: No se pudo calcular la latitud media para convertir el perímetro geográfico a km.")
                perimeter_km = np.nan
        else:
            # ASUME que las unidades proyectadas son METROS.
            perimeter_m = perimeter_native_units
            perimeter_km = perimeter_m / 1000.0
            print(f"Perímetro (proyectado): {perimeter_km:.3f} km (longitud nativa: {perimeter_native_units:.2f} unidades CRS)")

        return perimeter_km

    except Exception as e:
        print(f"Error calculando perímetro con shapes: {e}. Intentando fallback.")
        # Fallback: Aproximación por borde de píxeles (less accurate)
        try:
            eroded_mask = binary_erosion(watershed_mask)
            border_mask = watershed_mask ^ eroded_mask
            num_border_cells = np.sum(border_mask)

            # Calcular longitud promedio de lado de celda en metros
            res_x_native = abs(transform.a)
            res_y_native = abs(transform.e)
            avg_res_m = np.nan

            if is_geographic:
                mean_lat = None
                where_border = np.where(border_mask)
                if where_border[0].size > 0:
                     rows_in_border = where_border[0]
                     cols_in_border = where_border[1]
                     _, lats = rasterio.transform.xy(transform, rows_in_border, cols_in_border)
                     mean_lat = np.mean(lats)

                if mean_lat is not None:
                    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_lat)) + 1.175 * math.cos(4 * math.radians(mean_lat))
                    m_per_deg_lon = 111412.84 * math.cos(math.radians(mean_lat)) - 93.5 * math.cos(3 * math.radians(mean_lat)) + 0.118 * math.cos(5 * math.radians(mean_lat))
                    res_x_m = res_x_native * m_per_deg_lon
                    res_y_m = res_y_native * m_per_deg_lat
                    avg_res_m = (res_x_m + res_y_m) / 2.0
                else:
                    print("Advertencia: No se pudo calcular latitud media para fallback de perímetro.")
                    return np.nan # No se puede calcular sin latitud
            else:
                # ASUME unidades proyectadas en METROS
                avg_res_m = (res_x_native + res_y_native) / 2.0

            perimeter_m_approx = num_border_cells * avg_res_m
            perimeter_km_approx = perimeter_m_approx / 1000.0
            print(f"Perímetro (aprox. píxeles): {perimeter_km_approx:.3f} km")
            return perimeter_km_approx
        except Exception as e2:
            print(f"Error en fallback de cálculo de perímetro: {e2}")
            return np.nan

def calculate_morphometric_indices(data):
    global max_strahler_order
    """Calcula varios índices morfométricos a partir de los datos guardados."""
    ACLARA.config(state=tk.NORMAL)
    if not data["calculated"]:
        messagebox.showerror("Error", "No hay datos de cuenca calculados.")
        return None

    print("\n--- Calculando Índices Morfométricos ---")
    indices = {}

    # Datos básicos
    ws_mask = data["watershed_mask"]
    rivers_mask_display = data["rivers_mask"] # Enhanced rivers for Lt and Dd
    rivers_for_strahler_mask = data["rivers_for_strahler_mask"] # Original rivers for Strahler
    dem = data["dem_data"]
    inflated_dem = data["inflated_dem"]
    transform = data["transform"]
    is_geographic = data["is_geographic"]
    src_res = data["src_res"]
    pysheds_grid = data["pysheds_grid"] # Objeto Grid de PySheds
    outlet_row, outlet_col = data["outlet_coords"]
    fdir_np = data["fdir"]
    strahler_order_np = data["strahler_order_data"] # Datos del orden de Strahler

    # --- VERIFICACIÓN DE UNIDADES DE ENTRADA ---
    print(f"Asunciones: Unidades DEM = metros. Unidades CRS proyectado = metros.")
    if is_geographic:
        print("CRS es Geográfico (grados). Se usarán conversiones aproximadas a metros/km.")
    else:
        print("CRS es Proyectado. Se asumen unidades en metros.")
    # -------------------------------------------

    # 1. Área (A)
    area_km2 = calculate_area_km2(ws_mask, transform, is_geographic, src_res)
    indices["Área (A)"] = f"{area_km2:.3f} km²"
    if area_km2 <= 0:
        print("Error: Área de la cuenca es cero o negativa.")
        return {"Error": "Área de la cuenca es cero o negativa."}

    # 2. Perímetro (P)
    perimeter_km = calculate_perimeter_km(ws_mask, transform, is_geographic)
    indices["Perímetro (P)"] = f"{perimeter_km:.3f} km" if not np.isnan(perimeter_km) else "No calculado"

    # 3. Relieve de la Cuenca (H)
    dem_in_watershed = dem[ws_mask]
    valid_dem_in_watershed = dem_in_watershed[~np.isnan(dem_in_watershed) & np.isfinite(dem_in_watershed)]
    relief_h = np.nan
    if valid_dem_in_watershed.size > 0:
        elev_max = np.max(valid_dem_in_watershed)
        elev_min = np.min(valid_dem_in_watershed)
        relief_h = elev_max - elev_min
        indices["Elevación Máxima"] = f"{elev_max:.2f} m"
        indices["Elevación Mínima"] = f"{elev_min:.2f} m"
        indices["Relieve (H)"] = f"{relief_h:.2f} m"
        print(f"Relieve (H): {relief_h:.2f} m (Asumiendo DEM en metros)")
    else:
        indices["Elevación Máxima"] = "No calculado"
        indices["Elevación Mínima"] = "No calculado"
        indices["Relieve (H)"] = "No calculado"

    # 4. Pendiente Media de la Cuenca (S_avg)
    mean_slope_percent = np.nan
    try:
        print("Calculando pendiente media...")
        dy, dx = np.gradient(inflated_dem) # Cambio de elevación por celda (unidades DEM)

        res_x_native = abs(transform.a)
        res_y_native = abs(transform.e)
        dist_x_m = np.nan
        dist_y_m = np.nan

        if is_geographic:
            mean_lat = None
            where_watershed = np.where(ws_mask)
            if where_watershed[0].size > 0:
                rows_in_watershed = where_watershed[0]
                cols_in_watershed = where_watershed[1]
                _, lats = rasterio.transform.xy(transform, rows_in_watershed, cols_in_watershed)
                mean_lat = np.mean(lats)
            if mean_lat is not None:
                 m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_lat)) + 1.175 * math.cos(4 * math.radians(mean_lat))
                 m_per_deg_lon = 111412.84 * math.cos(math.radians(mean_lat)) - 93.5 * math.cos(3 * math.radians(mean_lat)) + 0.118 * math.cos(5 * math.radians(mean_lat))
                 dist_x_m = res_x_native * m_per_deg_lon
                 dist_y_m = res_y_native * m_per_deg_lat
            else:
                 print("Advertencia: No se pudo calcular latitud media para pendiente.")
                 raise ValueError("Latitud media no calculable para pendiente geográfica.")
        else:
            dist_x_m = res_x_native
            dist_y_m = res_y_native

        # Evitar división por cero si la resolución es cero
        if dist_x_m == 0 or dist_y_m == 0:
             raise ValueError("Resolución de celda cero detectada.")

        grad_y = dy / dist_y_m
        grad_x = dx / dist_x_m
        slope_m_m = np.sqrt(grad_x**2 + grad_y**2)
        slope_percent = slope_m_m * 100

        slope_in_watershed = slope_percent[ws_mask]
        valid_slope_in_watershed = slope_in_watershed[~np.isnan(slope_in_watershed) & np.isfinite(slope_in_watershed)]

        if valid_slope_in_watershed.size > 0:
            mean_slope_percent = np.mean(valid_slope_in_watershed)
            indices["Pendiente Media (S_avg)"] = f"{mean_slope_percent:.2f} %"
            print(f"Pendiente media calculada: {mean_slope_percent:.2f} %")
        else:
             print("Advertencia: No se encontraron valores de pendiente válidos en la cuenca.")
             indices["Pendiente Media (S_avg)"] = "No calculado (sin valores válidos)"

    except Exception as e:
        print(f"Error calculando pendiente media con gradiente: {e}. Intentando fallback.")
        if not np.isnan(relief_h) and relief_h > 0 and area_km2 > 0:
            area_m2 = area_km2 * 1_000_000
            char_length_m = math.sqrt(area_m2)
            if char_length_m > 0:
                mean_slope_m_m_approx = relief_h / char_length_m
                mean_slope_percent = mean_slope_m_m_approx * 100
                indices["Pendiente Media (S_avg)"] = f"{mean_slope_percent:.2f} % (Estimada H/√A)"
                print(f"Pendiente media (estimada): {mean_slope_percent:.2f} %")
            else:
                indices["Pendiente Media (S_avg)"] = "No calculado (fallback falló)"
        else:
            indices["Pendiente Media (S_avg)"] = f"No calculado (error o datos insuficientes)"

    # 5. Longitud del Cauce Principal (Lcp)
    lcp_km = np.nan
    path_coords = None # Guardará las coordenadas (filas, columnas) de la ruta

    try:
        print("Calculando longitud del cauce principal...")
        outlet_rc = (outlet_row, outlet_col) # Outlet coordinates in (row, col)
        print(f"DEBUG: Outlet para Lcp (fila, col): {outlet_rc}")

        if pysheds_grid is None:
             raise AttributeError("El objeto PySheds Grid es None.")
        if not hasattr(pysheds_grid, 'fdir') or pysheds_grid.fdir is None:
             raise AttributeError("El objeto PySheds Grid o su atributo 'fdir' no están disponibles para longest_flowpath.")

        fdir_val_at_outlet = np.nan # Valor por defecto
        try:
            if fdir_np is None:
                raise ValueError("El array numpy de dirección de flujo (fdir_np) no está disponible.")
            fdir_val_at_outlet = fdir_np[outlet_rc[0], outlet_rc[1]]
            print(f"DEBUG: Valor FDIR en outlet ({outlet_rc[0]}, {outlet_rc[1]}): {fdir_val_at_outlet}")
            if np.isnan(fdir_val_at_outlet) or fdir_val_at_outlet <= 0:
                 raise ValueError(f"Valor FDIR inválido ({fdir_val_at_outlet}) en el punto de salida {outlet_rc}. No se puede trazar la ruta.")
        except IndexError:
            print(f"Error: Coordenadas de salida {outlet_rc} fuera de los límites del array FDIR.")
            raise ValueError(f"Outlet {outlet_rc} fuera de límites FDIR.")
        except Exception as e_fdir_check:
            print(f"Advertencia: No se pudo verificar el valor FDIR en el outlet. Error: {e_fdir_check}")
            raise ValueError(f"No se pudo verificar FDIR en outlet {outlet_rc}.")

        print("DEBUG: Llamando a pysheds_grid.longest_flowpath...")
        path_coords_rc, path_dist_native_list = pysheds_grid.longest_flowpath(
            xy=outlet_rc,           # Outlet coordinates (row, col)
            fdir=pysheds_grid.fdir, # Usar el fdir almacenado en el objeto grid
            routing='d8'            # Routing algorithm
        )

        if path_coords_rc is not None and len(path_coords_rc[0]) > 1 and path_dist_native_list is not None and len(path_dist_native_list) > 0:
            print("DEBUG: longest_flowpath devolvió una ruta válida.")
            path_coords = path_coords_rc # Guardar coordenadas (filas, columnas) para cálculo de pendiente
            lcp_native_units = path_dist_native_list[-1] # La longitud total es el último valor de la lista de distancias

            # Convert native length to km
            if is_geographic:
                path_rows = path_coords[0]
                path_cols = path_coords[1]
                _, path_lats = rasterio.transform.xy(transform, path_rows, path_cols)
                mean_path_lat = np.mean(path_lats)

                m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_path_lat)) + 1.175 * math.cos(4 * math.radians(mean_path_lat))
                m_per_deg_lon = 111412.84 * math.cos(math.radians(mean_path_lat)) - 93.5 * math.cos(3 * math.radians(mean_path_lat)) + 0.118 * math.cos(5 * math.radians(mean_path_lat))
                avg_m_per_degree = (m_per_deg_lat + m_per_deg_lon) / 2.0

                lcp_m = lcp_native_units * avg_m_per_degree
                lcp_km = lcp_m / 1000.0
                print(f"Lcp (geográfico): {lcp_km:.3f} km (dist nativa: {lcp_native_units:.4f} grados, lat media cauce: {mean_path_lat:.4f})")
            else:
                lcp_m = lcp_native_units
                lcp_km = lcp_m / 1000.0
                print(f"Lcp (proyectado): {lcp_km:.3f} km (dist nativa: {lcp_native_units:.2f} m)")

            lcp_km = max(0.001, lcp_km) if not np.isnan(lcp_km) else np.nan

        else:
            print("Advertencia: pysheds_grid.longest_flowpath no devolvió una ruta válida (None, vacía, o longitud <= 1).")
            raise ValueError("longest_flowpath falló o devolvió resultado inválido")

    except AttributeError as ae:
         print(f"Error calculando Lcp: {ae}. Verifique si fdir se calculo correctamente y está en el objeto grid.")
         lcp_km = np.nan # Ensure lcp_km remains NaN
         if area_km2 > 0:
             lcp_km = 1.4 * math.sqrt(area_km2)
             print(f"Lcp (estimado por área): {lcp_km:.3f} km")
         else:
             lcp_km = np.nan
             print("No se puede estimar Lcp (área <= 0).")
    except ValueError as ve:
         print(f"Error calculando Lcp: {ve}. Intentando fallback con fórmula empírica.")
         if area_km2 > 0:
             lcp_km = 1.4 * math.sqrt(area_km2)
             print(f"Lcp (estimado por área): {lcp_km:.3f} km")
         else:
             lcp_km = np.nan
             print("No se puede estimar Lcp (área <= 0).")
    except Exception as e:
        print(f"Error inesperado calculando Lcp con pysheds ({e}). Intentando fallback con fórmula empírica.")
        traceback.print_exc()
        if area_km2 > 0:
             lcp_km = 1.4 * math.sqrt(area_km2)
             print(f"Lcp (estimado por área): {lcp_km:.3f} km")
        else:
             lcp_km = np.nan
             print("No se puede estimar Lcp (área <= 0).")

    # Assign the final value (or NaN) to the indices dictionary
    indices["Longitud Cauce Principal (Lcp)"] = f"{lcp_km:.3f} km" if not np.isnan(lcp_km) else "No calculado"

    # 5b. Pendiente del Cauce Principal
    slope_channel_percent = np.nan
    slope_channel_mm = np.nan
    try:
        print("Calculando pendiente del cauce principal...")
        if path_coords is not None and not np.isnan(lcp_km) and lcp_km > 0:
            path_rows = path_coords[0].astype(int)
            path_cols = path_coords[1].astype(int)

            valid_indices = (path_rows >= 0) & (path_rows < inflated_dem.shape[0]) & \
                            (path_cols >= 0) & (path_cols < inflated_dem.shape[1])

            elev_channel = inflated_dem[path_rows[valid_indices], path_cols[valid_indices]]

            valid_elev = elev_channel[~np.isnan(elev_channel) & np.isfinite(elev_channel)]

            if len(valid_elev) >= 2:
                elev_start_channel = valid_elev[0]
                elev_end_channel = valid_elev[-1]
                channel_relief = elev_start_channel - elev_end_channel

                print(f"DEBUG: Elevación inicio cauce: {elev_start_channel:.2f} m")
                print(f"DEBUG: Elevación fin cauce (outlet): {elev_end_channel:.2f} m")
                print(f"DEBUG: Relieve del cauce: {channel_relief:.2f} m")

                if channel_relief < 0:
                    print(f"Advertencia: Relieve del cauce negativo ({channel_relief:.2f} m). Usando valor absoluto.")
                    channel_relief = abs(channel_relief)
                elif channel_relief == 0:
                     print(f"Advertencia: Relieve del cauce es cero. La pendiente será cero.")


                lcp_m = lcp_km * 1000.0 # Convertir Lcp a metros
                if lcp_m > 0:
                    slope_channel_mm = channel_relief / lcp_m # Pendiente en m/m
                    slope_channel_percent = slope_channel_mm * 100 # Pendiente en %
                    indices["Pendiente Cauce Principal"] = f"{slope_channel_percent:.2f} %"
                    indices["Pendiente Cauce Principal (m/m)"] = f"{slope_channel_mm:.5f}"
                    print(f"Pendiente cauce principal: {slope_channel_percent:.2f} % ({slope_channel_mm:.5f} m/m)")
                else:
                    indices["Pendiente Cauce Principal"] = "No calculado (Lcp=0)"
                    indices["Pendiente Cauce Principal (m/m)"] = "No calculado (Lcp=0)"
                    print("Error: Lcp en metros es cero, no se puede calcular pendiente.")
            else:
                indices["Pendiente Cauce Principal"] = "No calculado (puntos insuficientes)"
                indices["Pendiente Cauce Principal (m/m)"] = "No calculado (puntos insuficientes)"
                print("Advertencia: No se encontraron suficientes puntos de elevación válidos a lo largo del cauce principal.")
        else:
            print("No se pudo calcular la pendiente del cauce directamente (falta ruta o Lcp). Intentando fallback H/Lcp.")
            if not np.isnan(relief_h) and relief_h >= 0 and not np.isnan(lcp_km) and lcp_km > 0:
                 lcp_m = lcp_km * 1000.0
                 slope_channel_mm = relief_h / lcp_m # Usa el relieve TOTAL de la cuenca
                 slope_channel_percent = slope_channel_mm * 100
                 indices["Pendiente Cauce Principal"] = f"{slope_channel_percent:.2f} % (Estimada H/Lcp)"
                 indices["Pendiente Cauce Principal (m/m)"] = f"{slope_channel_mm:.5f} (Estimada H/Lcp)"
                 print(f"Pendiente cauce principal (estimada H/Lcp): {slope_channel_percent:.2f} %")
            else:
                 indices["Pendiente Cauce Principal"] = "No calculado (sin ruta/Lcp/H)"
                 indices["Pendiente Cauce Principal (m/m)"] = "No calculado (sin ruta/Lcp/H)"
                 print("No se pudo estimar la pendiente del cauce (faltan H o Lcp).")

    except Exception as e:
        print(f"Error inesperado calculando pendiente del cauce principal: {e}")
        traceback.print_exc()
        indices["Pendiente Cauce Principal"] = f"Error"
        indices["Pendiente Cauce Principal (m/m)"] = f"Error"


    # 6. Longitud Total de Cauces (Lt) - Aproximación por celda
    lt_km = 0.0
    # Use the enhanced rivers_mask for Lt, as it represents the visually displayed network
    num_river_cells = np.sum(rivers_mask_display) 
    if num_river_cells > 0:
        res_x_native = abs(transform.a)
        res_y_native = abs(transform.e)
        avg_res_m = np.nan

        if is_geographic:
            mean_lat_rivers = None
            where_rivers = np.where(rivers_mask_display) # Use rivers_mask_display here
            if where_rivers[0].size > 0:
                rows_in_rivers = where_rivers[0]
                cols_in_rivers = where_rivers[1] # Corrected: cols_in_rivers = where_rivers[1]
                _, lats = rasterio.transform.xy(transform, rows_in_rivers, cols_in_rivers)
                mean_lat_rivers = np.mean(lats)

            if mean_lat_rivers is not None:
                m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_lat_rivers)) + 1.175 * math.cos(4 * math.radians(mean_lat_rivers))
                m_per_deg_lon = 111412.84 * math.cos(math.radians(mean_lat_rivers)) - 93.5 * math.cos(3 * math.radians(mean_lat_rivers)) + 0.118 * math.cos(5 * math.radians(mean_lat_rivers))
                res_x_m = res_x_native * m_per_deg_lon
                res_y_m = res_y_native * m_per_deg_lat
                # Usar longitud diagonal promedio de la celda
                avg_res_m = math.sqrt((res_x_m**2 + res_y_m**2) / 2)
            else:
                print("Advertencia: No se pudo calcular latitud media para Lt.")
                avg_res_m = 0
        else:
            # Asume unidades proyectadas en metros
            res_x_m = res_x_native
            res_y_m = res_y_native
            avg_res_m = math.sqrt((res_x_m**2 + res_y_m**2) / 2)

        lt_m = num_river_cells * avg_res_m
        lt_km = lt_m / 1000.0
        indices["Longitud Total Cauces (Lt, aprox.)"] = f"{lt_km:.3f} km"
        print(f"Longitud total cauces (aprox.): {lt_km:.3f} km ({num_river_cells} celdas de río)")
    else:
        indices["Longitud Total Cauces (Lt, aprox.)"] = "0.000 km"


    # 7. Densidad de Drenaje (Dd)
    dd = np.nan
    if area_km2 > 0 and not np.isnan(lt_km):
        dd = lt_km / area_km2 if area_km2 > 0 else 0.0
        indices["Densidad de Drenaje (Dd)"] = f"{dd:.3f} km/km²"
        print(f"Densidad de Drenaje (Dd): {dd:.3f} km/km²")
    else:
        indices["Densidad de Drenaje (Dd)"] = "No calculado"


    # 8. Factor de Forma (Kf)
    kf = np.nan
    if area_km2 > 0 and not np.isnan(lcp_km) and lcp_km > 0:
        kf = area_km2 / (lcp_km ** 2)
        indices["Factor de Forma (Kf)"] = f"{kf:.3f}"
        print(f"Factor de Forma (Kf): {kf:.3f}")
    else:
        indices["Factor de Forma (Kf)"] = "No calculado"


    # 9. Coeficiente de Compacidad (Kc) o Índice de Gravelius
    kc = np.nan
    if not np.isnan(perimeter_km) and perimeter_km > 0 and area_km2 > 0:
        denominator = 2 * math.sqrt(math.pi * area_km2)
        kc = perimeter_km / denominator if denominator > 0 else np.nan
        indices["Coeficiente de Compacidad (Kc)"] = f"{kc:.3f}" if not np.isnan(kc) else "No calculado (denominador cero)"
        print(f"Coeficiente de Compacidad (Kc): {kc:.3f}" if not np.isnan(kc) else "No calculado")
    else:
        indices["Coeficiente de Compacidad (Kc)"] = "No calculado"

    # 10. Estadísticas de Orden de Strahler (NUEVO)
    # Use the specific mask for Strahler
    if strahler_order_np is not None and np.any(rivers_for_strahler_mask): 
        strahler_in_watershed = strahler_order_np[rivers_for_strahler_mask] # Use the specific mask
        valid_strahler = strahler_in_watershed[~np.isnan(strahler_in_watershed) & (strahler_in_watershed > 0)]
        if valid_strahler.size > 0:
            max_strahler_order = int(np.max(valid_strahler))
            indices["Orden de Strahler Máximo"] = f"{max_strahler_order}"
            print(f"Orden de Strahler Máximo: {max_strahler_order}")
            # Contar número de segmentos por orden
            order_counts = {}
            for order_val in np.unique(valid_strahler):
                if order_val > 0:
                    order_counts[int(order_val)] = np.sum(valid_strahler == order_val)
            
            # Formatear para mostrar
            order_counts_str = ", ".join([f"Orden {o}: {c} seg." for o, c in sorted(order_counts.items())])
            indices["Segmentos por Orden"] = order_counts_str
            print(f"Segmentos por Orden: {order_counts_str}")
        else:
            indices["Orden de Strahler Máximo"] = "No calculado (sin ríos válidos)"
            indices["Segmentos por Orden"] = "No calculado (sin ríos válidos)"
    else:
        indices["Orden de Strahler Máximo"] = "No calculado (datos no disponibles)"
        indices["Segmentos por Orden"] = "No calculado (datos no disponibles)"


    print("--- Cálculo de Índices Morfométricos Finalizado ---")
    return indices

# --- Funciones de Interfaz Gráfica y Eventos ---

def onclick(event):
    """Manejador de clics en el mapa: guarda coordenadas y cierra la figura."""
    global clicked_coords
    # Verificar si el clic fue dentro de los ejes del gráfico
    # Y si no está activo algún modo de la toolbar (zoom, pan)
    if event.xdata is not None and event.ydata is not None \
       and event.inaxes == event.canvas.figure.axes[0] \
       and event.canvas.toolbar.mode == '': # Check if toolbar mode is inactive
        clicked_coords[0] = event.xdata
        clicked_coords[1] = event.ydata
        print(f"Punto seleccionado en el mapa: X={clicked_coords[0]:.4f}, Y={clicked_coords[1]:.4f}")
        # Cerrar la figura donde se hizo clic
        plt.close(event.canvas.figure)
    elif event.canvas.toolbar.mode != '':
        print(f"Modo de toolbar activo ({event.canvas.toolbar.mode}), clic ignorado para selección de punto.")
    else:
        print("Clic fuera de los ejes del mapa.")

def ask_manual_coordinates(parent_window):
    """Pide al usuario las coordenadas X e Y manualmente."""
    x_coord_str = simpledialog.askstring("Coordenada X", "Ingrese la coordenada X (Longitud) del punto de salida:", parent=parent_window)
    if x_coord_str is None: return None, None # Cancelado

    y_coord_str = simpledialog.askstring("Coordenada Y", "Ingrese la coordenada Y (Latitud) del punto de salida:", parent=parent_window)
    if y_coord_str is None: return None, None # Cancelado

    try:
        x_coord = float(x_coord_str)
        y_coord = float(y_coord_str)
        return x_coord, y_coord
    except (ValueError, TypeError):
        messagebox.showerror("Error de Entrada", "Las coordenadas ingresadas no son números válidos.", parent=parent_window)
        return None, None
def ADIOS():
    top.destroy()
def AVISA():
    global top
    top = Toplevel(root_window)
    top.title("AVISO ---> RESOLUCIÓN DE LOS MDE")
    top.geometry("450x400+100+100")
    RECOMIENDA= Label(top, text="                                  A C U A C   C U E N C A S\n\
    \n\
    SE RECOMIENDA UTILIZAR MODELOS DIGITALES DE ELEVACION\n\
    (MDE) A NIVEL MUNICIPAL. LA APP NO DESARROLLARÁ BIEN\n\
    CON MDE MAYORES A 10,000 Has.\n\
    LOS MDE A NIVEL MUNICIPAL, SE PUEDEN OBTENER EN:\n\
    https://www.inegi.org.mx/app/mapas/default.html?t=193&ag=32\n\
    PARA CUALQUIER MUNICIPIO DEL PAIS\n\
    LOS DEM EN ESA LIGA, NO MUESTRAN LA\n\
    FORMA (SHAPE) DEL MUNICIPIO, SI ESE\n\
    FUERA EL INTERÉS, TENDRÁ QUE SER \n\
    RECORTADO UTILIZANDO LOS PROCESOS INDICADOS\n\
    EN CUALQUIER SIG PARA POSETERIORMENTE SER \n\
    UTILIZADOS EN ESTA APLICACIÓN", justify="left",fg="blue")
    RECOMIENDA.place(x=50, y=50)
    BY= Button(top, text= "SALIR", command= ADIOS)
    BY.place(x= 200, y=300)
    ACLARA.config(state=tk.NORMAL)
    pass
def abrir():
    import subprocess
    try:
        # Intenta abrir el archivo .xlsm directamente
        subprocess.Popen(["start", "APLICACION.xlsm"], shell=True)
    except FileNotFoundError:
        messagebox.showerror("Error", "No se encontró el archivo 'APLICACION.xlsm'. Asegúrese de que esté en la misma carpeta que el script.")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo abrir el archivo Excel:\n{e}")

def juimonos():
    top.destroy()
def ADIOS_STRAHLER():
    top.destroy()
def INTERPRETA():
    global top
    global ADIOS
    global max_strahler_order # Asegurarse de que la variable global esté disponible

    top = Toplevel(root_window)
    top.title("INTERPRETACION DEL ORDEN DE STRAHLER")
    top.geometry("850x550+100+100") # Aumentar altura para más texto
    
    # Título general
    Label(top, text="INTERPRETACIÓN DEL ORDEN DE STRAHLER",
          font=("Comic Sans MS", 14, "bold"), fg="blue").pack(pady=10)
    
    # Introducción
    Label(top, text="El orden de Strahler es un indicador clave de la complejidad y madurez geomorfológica de una cuenca, así como de su respuesta hidrológica ante eventos de precipitación.",
          font=("Comic Sans MS", 10), fg="black", justify="left", wraplength=750).pack(pady=(0, 10), padx=20)

    if max_strahler_order is not None:
        order = int(max_strahler_order)
        
        interpretation_text = ""
        if order == 1:
            interpretation_text = """
            Orden 1: Corresponde a los cauces más pequeños, sin afluentes. Son las cabeceras de la red de drenaje, a menudo efímeras o intermitentes, con pendientes pronunciadas. Son extremadamente sensibles a cambios locales como la deforestación o la erosión, y su respuesta hidrológica es rápida y localizada.
            """
        elif order == 2:
            interpretation_text = """
            Orden 2: Se forma cuando dos cauces de orden 1 se unen. Estos cauces son aún pequeños, pero ya muestran una mayor capacidad de transporte de agua y sedimentos. La cuenca es todavía muy influenciada por procesos locales y la respuesta hidrológica sigue siendo relativamente rápida.
            """
        elif order == 3:
            interpretation_text = """
            Orden 3: Resulta de la unión de dos cauces de orden 2. Representa un sistema de drenaje más desarrollado, con un flujo más constante. La cuenca comienza a integrar procesos de áreas más grandes, y la respuesta hidrológica es moderada, con un tiempo de concentración mayor que en órdenes inferiores.
            """
        elif order == 4:
            interpretation_text = """
            Orden 4: Se forma por la unión de dos cauces de orden 3. Indica una cuenca con una red de drenaje bien establecida y un cauce principal que ya drena una superficie considerable. El flujo es generalmente perenne y la cuenca tiene una capacidad significativa para generar escurrimiento, con una respuesta hidrológica más amortiguada.
            """
        elif order >= 5: # Agrupar 5 y superiores
            interpretation_text = f"""
            Orden {order} (o superior): Una cuenca con un orden de Strahler de 5 o más indica un sistema de drenaje grande y complejo, característico de ríos principales. Estos cauces integran el escurrimiento de vastas áreas, y su hidrología refleja las condiciones acumuladas de toda la cuenca. Son menos sensibles a impactos locales específicos, pero su salud depende de las prácticas de manejo en todos los tramos superiores. Se espera un escurrimiento cuantioso y una respuesta hidrológica más lenta y sostenida.
            """
        else:
            interpretation_text = """
            No se pudo determinar un orden de Strahler válido para la interpretación. Esto puede deberse a que la red de drenaje es extremadamente pequeña o a problemas en el cálculo.
            """
        
        Label(top, text=f"Orden de Strahler Máximo de la Cuenca: {order}\n\n{interpretation_text}",
              font=("Comic Sans MS", 10), fg="darkgreen", justify="left", wraplength=750).pack(pady=(10, 20), padx=20)
    else:
        Label(top, text="No se pudo determinar el orden de Strahler máximo para la interpretación.",
              font=("Comic Sans MS", 10), fg="red", justify="left").pack(pady=(10, 20), padx=20)

    ADIOS= Button(top, text= "SALIR", command= ADIOS_STRAHLER)
    ADIOS.pack(pady=10) # Usar pack para centrarlo mejor
        
def CONSIDERA():
    global top
    top = Toplevel()
    top.title("AVISO ---> CONSIDERACIONES DEL HIDROGRAMA ADIMENSIONAL")
    top.geometry("850x500+100+100")
    
                         
    ADIOS= Button(top, text= "SALIR", command= juimonos)
    ADIOS.place(x= 200, y=450)
    CONSIDERACIONES=Label(top, text='                                                                   A C U A C  C U E N C A S \n\
                                        Asunciones del Hidrograma Adimensional del SCS\n\
                                        \n\
    1.	La cuenca responde de manera lineal\n\
        o	El escurrimiento es proporcional al exceso de precipitación.\n\
        o	Si el volumen de exceso de lluvia se duplica, el hidrograma también lo hace.\n\
    2.	El exceso de precipitación se produce uniformemente en toda la cuenca\n\
        o	Se asume una distribución espacial uniforme de la lluvia efectiva (o exceso de precipitación).\n\
    3.	La duración del exceso de lluvia es corta y constante\n\
        o	Generalmente se toma un periodo de 1 hora o menos. El hidrograma generado corresponde a un pulso de lluvia de duración constante.\n\
    4.	El tiempo al pico y la forma del hidrograma pueden representarse de forma adimensional\n\
        o	SCS desarrolló un hidrograma con valores adimensionales de caudal y tiempo, aplicables a cualquier cuenca con ajustes simples.\n\
    5.	El hidrograma se puede escalar a diferentes condiciones\n\
        o	Una vez conocido el volumen de exceso de lluvia y el tiempo al pico, el hidrograma adimensional puede escalarse para obtener el hidrograma real.\n\
    6.	El escurrimiento base no se considera (en la versión estándar)\n\
        o	Solo representa el escurrimiento directo por exceso de precipitación. El flujo base debe sumarse si es necesario.\n\
    7.	El tiempo de concentración controla la respuesta de la cuenca\n\
        o	El tiempo al pico del hidrograma está relacionado directamente con el tiempo de concentración de la cuenca.\n\
    8.	El terreno tiene características hidrológicas homogéneas\n\
        o	Tipo de suelo, uso del suelo y condiciones de humedad similares en toda la cuenca.\n\
    9.          Para mejores resultados, la superficie de la cuenca deberá ser menor a 5000 km2, ó 500 mil ha.',fg="blue", justify="left"
        )
    CONSIDERACIONES.place(x=50, y=80)

def create_main_window(root, main_callback, load_existing_callback, manual_coord_callback, morph_callback):
    """Crea la ventana principal con opciones y el slider de umbral."""
    global morphometry_button, runoff_button, hydrograph_button,SIMULADOR
    global ACLARA
    global STRAHLER
    global STRAHLER_MAP # Nuevo botón para el mapa detallado

    root.title("PROYECTO ACUAC: APP PARA DELINEAR MICROCUENCAS")
    root.geometry("1050x850")
    frame = Frame(root, padx=20, pady=20)
    frame.pack(expand=True, fill="both")

    Label(frame, text="Delimitación de Micro - Cuencas\n A mayor resolución del DEM mejor la red de drenaje", font=("Arial", 12, "bold"),fg="blue").pack(pady=10)

    # --- Sección para Nueva Cuenca (con Slider) ---
    new_basin_frame = Frame(frame, pady=5)
    new_basin_frame.pack(fill="x")

    Label(new_basin_frame, text="Umbral de Acumulación para Red de Drenaje:\nMenor = mayor definición", font=("Arial", 10), justify="center").pack()
    threshold_var = IntVar(value=100)
    scale = Scale(new_basin_frame, from_=5, to=500, orient="horizontal", length=300,
                  variable=threshold_var, label="Umbral (celdas)")
    scale.pack(pady=5)

    # Botón para delinear haciendo clic en el mapa
    Button(new_basin_frame, text="Delinear Nueva Cuenca (Clic en Mapa)\n Se requiere el DEM tiff",
           command=lambda: main_callback(threshold_var.get(), use_manual_coords=False, manual_coords=None),
           width=35, height=2,bg="green",fg="white").pack(pady=5)
    Button(new_basin_frame,text= "IMPORTANTE", command=AVISA,width=10, height=1, fg="blue", bg="yellow").pack(pady=3)
    

    # Botón para delinear introduciendo coordenadas manualmente
    Button(new_basin_frame, text="Delinear Nueva Cuenca \n(Introduzca sus coordenadas X= Lon(-) Y= Lat)",
           command=lambda: manual_coord_callback(threshold_var.get()), # Llama a la función intermedia
           width=35, height=2,bg="green",fg="white").pack(pady=5)

    # --- Botón para Índices Morfométricos ---
    morphometry_button = Button(frame, text="Calcular Índices Morfométricos",
                                command=morph_callback, # Llama a show_morphometric_indices
                                width=35, height=2, state=tk.DISABLED) # Empieza deshabilitado
    morphometry_button.pack(pady=10)

    # --- Botón para Calcular Escurrimiento ---
    runoff_button = Button(frame, text="Calcular Escurrimiento (Método SCS-CN)",
                           command=calculate_runoff,
                           width=35, height=2, state=tk.DISABLED)
    runoff_button.pack(pady=10)

    # --- Botón para Hidrograma Adimensional SCS ---
    hydrograph_button = Button(frame, text="Generar Hidrograma Unitario SCS", # Nombre más preciso
                              command=generate_scs_hydrograph,
                              width=35, height=2, state=tk.DISABLED)
    hydrograph_button.pack(pady=10)
    ACLARA = Button(frame, text="COSIDERACIONES HU", command=CONSIDERA,
                    width=35, height=2, state=tk.DISABLED,fg="blue", bg="yellow")
    ACLARA.place(x= 700, y=440)

    STRAHLER = Button(frame, text="INTERPRETACIÓN STRAHLER", command =INTERPRETA,
                     width=35, height=2, state=tk.DISABLED, fg="blue", bg="yellow")
    STRAHLER.place(x=700, y=340)

    # Nuevo botón para mostrar el mapa detallado de Strahler
    STRAHLER_MAP = Button(frame, text="MAPA DETALLADO STRAHLER", command=show_strahler_map,
                         width=35, height=2, state=tk.DISABLED, fg="white", bg="blue")
    STRAHLER_MAP.place(x=700, y=390)  # Posicionado debajo del botón STRAHLER

    # --- Sección para Cargar Cuenca Existente y Ayuda ---
    
    other_buttons_frame = Frame(frame, pady=5)
    other_buttons_frame.pack(fill="x")
    SIMULADOR =Button(other_buttons_frame, text="SIMULADOR BALANCE DE AGUA", # Habilitado
            command=abrir,
           width=35, height=2,fg="white",bg="green").pack(pady=5)
    Button(other_buttons_frame, text="LEEME", command= helpiosa, width=35, height=2,bg="green",fg="white").pack(pady=5)

    # --- Botón de Salir ---
    Button(frame, text="SALIR", command=root.destroy, width=35, height=2,bg="green",fg="white").pack(pady=10)
    Label(frame, text="Desarrollado para análisis de microcuencas", font=("Arial", 8)).pack(side="bottom", pady=5)

def helpiosa():
    global top20
    top20=Toplevel()
    top20.title("Ayuda - Delineador de Cuencas")
    top20.geometry("500x500+50+50") # Ajustar tamaño y hacerla más alta
    top20.resizable(True, True) # Hacerla redimensionable para ver todo el texto

    help_text = """
                            P R O Y E C T O   A C U A C
                    
    EL CÓDIGO ORIGINAL HA SIDO PROGRAMADO EN PYTHON.
    CON ASESORÍA DE IA, SE HAN REPROGRAMADO LOS ALGORITMOS
    GEOESPACIALES.
    SEGUIR LOS SIGUIENTES PASOS:
    NO SE RECOMIENDA PARA CUENCAS GRANDES (> 500 MIL Ha.)
    El objetivo es delinear cuencas pequeñas o microcuencas que drenan a
    un punto de interés. Muy útil en captación de agua de lluvia.

    1.  Seleccione el Umbral de Acumulación:
        NÚMERO DE CELDAS AGUAS ARRIBA DEL
        PUNTO PARA CONSIDERAR QUE ES UN CAUCE.
        - Use el slider para definir la densidad de la red de drenaje.
        - Umbral bajo (~10): Red muy densa (más arroyos pequeños).
        - Umbral alto (~500): Red menos densa (ríos principales).
        - La visualización de la red puede variar según el umbral y la
          resolución del DEM.

    2.  Elija el Método de Delineación:
        a) Clic en Mapa:
           - Haga clic en "Delinear Nueva Cuenca (Clic en Mapa)".
           - Seleccione su archivo DEM (formato GeoTIFF).
           - Se mostrará un mapa interactivo con el DEM y la red de drenaje.
           - **¡IMPORTANTE! USO DEL ZOOM Y PAN:**
             Antes de hacer clic para seleccionar el punto de salida, use
             las herramientas de la barra de navegación de Matplotlib
             (generalmente en la parte inferior o superior de la ventana del gráfico):
               - **Zoom (Lupa):** Haga clic en el icono de la lupa. Luego,
                 haga clic y arrastre el mouse sobre el mapa para dibujar
                 un rectángulo. El mapa se acercará a esa área. Puede hacer
                 zoom repetidamente. Use los botones de flecha (izquierda/derecha)
                 en la barra para deshacer/recrear vistas.
               - **Pan (Flechas Cruzadas):** Haga clic en el icono de las
                 flechas cruzadas. Luego, haga clic y arrastre el mouse sobre
                 el mapa para mover la vista (desplazarse).
               - **Desactivar Zoom/Pan:** Haga clic nuevamente en el icono
                 activo (lupa o flechas) para desactivarlo ANTES de hacer
                 clic para seleccionar el punto.
           - Una vez que haya hecho zoom y/o pan para ver claramente el cauce
             deseado, **asegúrese de que ninguna herramienta (zoom/pan) esté activa**
             y haga clic directamente sobre la línea azul del cauce en el punto
             exacto de salida. El programa ajustará el clic al cauce más cercano
             con mayor acumulación.

        b) Coordenadas Manuales: Si conoce las coordenadas del punto
           - Haga clic en "Delinear Nueva Cuenca (Coords Manuales)".
           - Seleccione su archivo DEM (formato GeoTIFF).
           - Ingrese las coordenadas X (Longitud) e Y (Latitud) del punto
             de salida en las ventanas emergentes. Asegúrese de que las
             coordenadas estén dentro del área cubierta por su DEM y en el
             mismo Sistema de Coordenadas de Referencia (CRS).

    3.  Resultados:
        - Se mostrará un mapa con la cuenca delineada, la red de drenaje
          dentro de ella y el punto de salida ajustado (marcado en rojo).
        - Se calculará y mostrará el área de la cuenca.
        - Opcionalmente, puede guardar la cuenca como un archivo GeoTIFF.
        - Después de delinear, se habilitarán los botones para calcular
          índices morfométricos, escurrimiento e hidrograma.

    IMPORTANTE SOBRE UNIDADES Y COORDENADAS:
    - El programa ASUME que las unidades de elevación del DEM son METROS.
    - Si el DEM usa un Sistema de Coordenadas Proyectado (ej. UTM),
      el programa ASUME que las unidades de distancia son METROS.
    - Si el DEM usa Coordenadas Geográficas (Latitud/Longitud en grados),
      el programa realizará conversiones APROXIMADAS a metros/km, lo cual
      puede introducir imprecisiones, especialmente en áreas grandes.
    - Verifique siempre las unidades y el CRS de su DEM de entrada.
      Resultados incorrectos (muy altos o bajos) pueden deberse a
      unidades de entrada incorrectas (ej. pies en lugar de metros).
    - Problemas de 'nodata': Asegúrese que el valor nodata esté correctamente
      definido en su archivo GeoTIFF. El programa intenta leerlo, pero
      inconsistencias pueden causar errores.
    """
    
    text_frame = Frame(top20)
    text_frame.pack(expand=True, fill="both")

    text_area = Text(text_frame, wrap=tk.WORD, padx=10, pady=10, font=("Arial", 10))
    text_area.insert(tk.END, help_text)
    text_area.config(state=tk.DISABLED) # Hacer el texto no editable

    scrollbar = Scrollbar(text_frame, command=text_area.yview)
    text_area.config(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill="y")
    text_area.pack(side=tk.LEFT, expand=True, fill="both")

    # Botón para cerrar la ventana de ayuda
    close_button = Button(top20, text="Cerrar", command=top20.destroy)
    close_button.pack(pady=5)


def byby(): # Esta función no parece ser llamada, pero se deja por si acaso
    if 'top20' in globals() and top20.winfo_exists():
        top20.destroy()

def show_morphometric_indices():
    STRAHLER['state'] = 'normal'
    STRAHLER_MAP['state'] = 'normal' # Habilitar el nuevo botón del mapa de Strahler
    """Calcula y muestra los índices morfométricos en una nueva ventana."""
    global last_watershed_data, root_window

    if not last_watershed_data["calculated"]:
        messagebox.showinfo("Información", "Primero debe delinear una cuenca para calcular sus índices.", parent=root_window)
        return

    # Calcular índices
    indices = calculate_morphometric_indices(last_watershed_data)

    if indices is None:
         messagebox.showerror("Error", "No se pudieron calcular los índices morfométricos (función devolvió None).", parent=root_window)
         return
    if "Error" in indices:
        messagebox.showerror("Error", f"No se pudieron calcular los índices morfométricos.\n{indices.get('Error', '')}", parent=root_window)
        return

    # Crear ventana Toplevel para mostrar resultados
    top = Toplevel(root_window)
    top.title("Índices Morfométricos de la Cuenca")
    top.geometry("550x500+100+100") # Tamaño y posición (más ancho)

    # Crear un frame y un área de texto con scrollbar
    frame = Frame(top)
    frame.pack(expand=True, fill="both", padx=10, pady=10)

    text_area = Text(frame, wrap=tk.WORD, font=("Courier New", 10), height=25) # Usar fuente monoespaciada
    scrollbar = Scrollbar(frame, command=text_area.yview)
    text_area.config(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill="y")
    text_area.pack(side=tk.LEFT, expand=True, fill="both")

    # Formatear y añadir resultados al área de texto
    results_text = "--- Índices Morfométricos ---\n\n"
    results_text += "NOTA: Se asume que el DEM está en metros y los CRS proyectados en metros.\n"
    results_text += "      Las conversiones desde grados geográficos son aproximadas.\n\n"

    max_key_len = max(len(k) for k in indices.keys()) if indices else 0

    key_order = [
        "Área (A)", "Perímetro (P)",
        "Elevación Máxima", "Elevación Mínima", "Relieve (H)",
        "Pendiente Media (S_avg)",
        "Longitud Cauce Principal (Lcp)",
        "Pendiente Cauce Principal", "Pendiente Cauce Principal (m/m)",
        "Longitud Total Cauces (Lt, aprox.)",
        "Densidad de Drenaje (Dd)","Factor de Forma (Kf)", "Coeficiente de Compacidad (Kc)",
        "Orden de Strahler Máximo", "Segmentos por Orden" # NUEVO
    ]

    added_keys = set()
    for key in key_order:
        if key in indices:
            value = indices[key]
            padding = " " * (max_key_len - len(key) + 2)
            results_text += f"{key}:{padding}{value}\n"
            added_keys.add(key)

    # Añadir claves restantes que no estaban en el orden preferido
    for key, value in indices.items():
        if key not in added_keys:
            padding = " " * (max_key_len - len(key) + 2)
            results_text += f"{key}:{padding}{value}\n"


    text_area.insert(tk.END, results_text)
    text_area.config(state=tk.DISABLED)

    close_button = Button(top, text="Cerrar", command=top.destroy)
    close_button.pack(pady=5)

    top.transient(root_window)
    top.grab_set()
    root_window.wait_window(top)
def ayuda_CN():
    import subprocess
    try:
        subprocess.Popen(["start", "AYUDA_CN2.pdf"], shell=True)
    except FileNotFoundError:
        messagebox.showerror("Error", "No se encontró el archivo 'AYUDA_CN2.pdf'. Asegúrese de que esté en la misma carpeta que el script.")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo abrir el archivo PDF:\n{e}")
    
def calculate_runoff():
    """Calcula el escurrimiento para la cuenca delineada usando el método SCS-CN."""
    global last_watershed_data, root_window, p_event, cn # Hacer cn global

    if not last_watershed_data["calculated"]:
        messagebox.showinfo("Información", "Primero debe delinear una cuenca para calcular el escurrimiento.",
                            parent=root_window)
        return

    runoff_window = Toplevel(root_window)
    runoff_window.title("Cálculo de Escurrimiento - Método SCS-CN")
    runoff_window.geometry("400x350+100+100")
    runoff_window.transient(root_window)
    runoff_window.grab_set()

    main_frame = Frame(runoff_window, padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)

    Label(main_frame, text="Cálculo de Escurrimiento - Método SCS-CN",
          font=("Arial", 12, "bold"),fg="blue").pack(pady=(0, 10))
    Label(main_frame, text="Calcula la lámina y volumen de escurrimiento\ndirecto para una tormenta dada.",
          justify="center").pack(pady=(0, 15))

    input_frame = Frame(main_frame)
    input_frame.pack(fill="x", pady=10)

    cn_frame = Frame(input_frame)
    cn_frame.pack(fill="x", pady=5)
    Label(cn_frame, text="Número de Curva (CN):", width=25, anchor="w").pack(side="left")
    cn_var = tk.StringVar(value=" ") # Inicializar vacío o con valor por defecto si existe
    # Intentar usar valor previo si existe
    if 'cn' in globals() and cn is not None:
        cn_var.set(str(cn))
    cn_entry = tk.Entry(cn_frame, textvariable=cn_var, width=10)
    cn_entry.pack(side="left")
    Label(cn_frame, text="(1-100)", width=10).pack(side="left")

    p_frame = Frame(input_frame)
    p_frame.pack(fill="x", pady=5)
    Label(p_frame, text="Precipitación Total (P):", width=25, anchor="w").pack(side="left")
    p_var = tk.StringVar(value=" ") # Inicializar vacío o con valor por defecto si existe
    # Intentar usar valor previo si existe
    if 'p_event' in globals() and p_event is not None:
        p_var.set(str(p_event))
    p_entry = tk.Entry(p_frame, textvariable=p_var, width=10)
    p_entry.pack(side="left")
    Label(p_frame, text="(mm)", width=10).pack(side="left")

    result_frame = Frame(main_frame)
    result_frame.pack(fill="x", pady=10)
    result_label = Label(result_frame, text="", font=("Arial", 10, "bold"), justify="left")
    result_label.pack(pady=10, anchor="w")

    # Inicializar p_event y cn si no existen
    if 'p_event' not in globals():
        p_event = None
    if 'cn' not in globals():
        cn = None

    def calculate_and_show():
        global p_event, cn # Asegurar que se modifican las globales
        global HELP_CN
        try:
            cn_val = float(cn_var.get())
            p_val = float(p_var.get())

            if not (1 <= cn_val <= 100):
                messagebox.showerror("Error", "El Número de Curva (CN) debe estar entre 1 y 100.",
                                     parent=runoff_window)
                return
            if p_val < 0:
                messagebox.showerror("Error", "La precipitación no puede ser negativa.",
                                     parent=runoff_window)
                return

            # Guardar los valores válidos en las variables globales
            cn = cn_val
            p_event = p_val

            s = (25400 / cn) - 254
            ia = 0.2 * s
            q = 0.0
            if p_event > ia:
                q = ((p_event - ia) ** 2) / (p_event - ia + s)

            area_km2 = calculate_area_km2(last_watershed_data["watershed_mask"],
                                         last_watershed_data["transform"],
                                         last_watershed_data["is_geographic"])

            if area_km2 <= 0:
                 messagebox.showerror("Error", "El área de la cuenca es cero o inválida.",
                                     parent=runoff_window)
                 result_label.config(text="Error: Área de cuenca inválida")
                 return

            volume_m3 = (area_km2 * 1_000_000) * (q / 1000.0)

            results_text = f"Lámina Escurrimiento (Q): {q:.2f} mm\n"
            results_text += f"Volumen Total: {volume_m3:.2f} m³\n"
            results_text += f"Volumen Total: {volume_m3 / 1_000_000:.4f} hm³ (millones de m³)"
            result_label.config(text=results_text)

        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos para CN y P.",
                                 parent=runoff_window)
            result_label.config(text="Error en la entrada.")
        except Exception as e:
             messagebox.showerror("Error", f"Ocurrió un error inesperado:\n{e}",
                                 parent=runoff_window)
             result_label.config(text="Error inesperado.")

    button_frame = Frame(main_frame)
    button_frame.pack(fill="x", pady=10)
    calculate_button = Button(button_frame, text="Calcular", command=calculate_and_show, width=15)
    calculate_button.pack(side="left", padx=10)
    close_button = Button(button_frame, text="Cerrar", command=runoff_window.destroy, width=15)
    close_button.pack(side="right", padx=10)

    HELP_CN= Button(button_frame, text="Ayuda CN", command=ayuda_CN)
    HELP_CN.pack(side="bottom", padx=10)

    runoff_window.update_idletasks()
    w = runoff_window.winfo_width()
    h = runoff_window.winfo_height()
    x = (runoff_window.winfo_screenwidth() // 2) - (w // 2)
    y = (runoff_window.winfo_screenheight() // 2) - (h // 2)
    runoff_window.geometry(f'{w}x{h}+{x}+{y}')
    root_window.wait_window(runoff_window)

def generate_scs_hydrograph():
    """Genera y muestra el hidrograma unitario del método SCS."""
    global last_watershed_data, root_window, p_event, cn # Usar cn global

    if not last_watershed_data["calculated"]:
        messagebox.showinfo("Información", "Primero debe delinear una cuenca para generar el hidrograma.",
                            parent=root_window)
        return

    area_km2 = np.nan
    lcp_km = np.nan
    slope_channel_mm = np.nan
    tc_min = np.nan
    default_cn = 75 # Valor por defecto si no hay cálculo previo
    default_p = 50  # Valor por defecto si no hay cálculo previo
    default_d = 1   # Duración por defecto
    default_lcp = 0.0  # Valor por defecto para longitud del cauce principal manual

    # Intentar obtener valores de cálculos previos
    if 'cn' in globals() and cn is not None:
        default_cn = cn
        print(f"Usando CN del cálculo anterior: {default_cn}")
    if 'p_event' in globals() and p_event is not None:
        default_p = p_event
        print(f"Usando precipitación (P) del cálculo anterior: {default_p:.1f} mm")

    try:
        # Calcular área (necesaria siempre)
        area_km2 = calculate_area_km2(
            last_watershed_data["watershed_mask"],
            last_watershed_data["transform"],
            last_watershed_data["is_geographic"]
        )
        if area_km2 <= 0:
            raise ValueError("Área de la cuenca es cero o inválida.")

        # Calcular índices para obtener Lcp y Pendiente del cauce
        indices = calculate_morphometric_indices(last_watershed_data)
        if indices is None or "Error" in indices:
            print("Advertencia: No se pudieron calcular índices morfométricos para Tc. Usando valor por defecto.")
            # No lanzar error aquí, permitir continuar con Tc por defecto
        else:
            # Extraer Lcp
            lcp_str = indices.get("Longitud Cauce Principal (Lcp)", "nan km")
            try:
                if "No calculado" not in lcp_str and "Error" not in lcp_str:
                    lcp_km = float(lcp_str.split()[0])
                    # Usar longitud calculada como valor por defecto para la manual
                    default_lcp = lcp_km
                else:
                    lcp_km = np.nan
            except:
                lcp_km = np.nan
                print("Advertencia: No se pudo extraer Lcp numérico de los índices.")

            # Extraer Pendiente del cauce (m/m)
            slope_mm_str = indices.get("Pendiente Cauce Principal (m/m)", "nan")
            try:
                if "No calculado" not in slope_mm_str and "Error" not in slope_mm_str:
                    # Usar regex para extraer el número flotante de forma más robusta
                    match = re.search(r"[-+]?\d*\.\d+|\d+", slope_mm_str)
                    if match:
                        slope_channel_mm = float(match.group(0))
                    else:
                        slope_channel_mm = np.nan
                else:
                     slope_channel_mm = np.nan
            except:
                slope_channel_mm = np.nan
                print("Advertencia: No se pudo extraer la pendiente del cauce (m/m) numérica de los índices.")

            # Calcular Tc con Kirpich si hay datos
            if not np.isnan(lcp_km) and lcp_km > 0 and not np.isnan(slope_channel_mm) and slope_channel_mm > 0:
                lcp_m = lcp_km * 1000.0
                # Asegurar que la pendiente no sea cero para evitar división por cero o exponente negativo grande
                slope_channel_mm_safe = max(slope_channel_mm, 0.00001)
                # Fórmula de Kirpich (Tc en minutos)
                tc_min = 0.01947 * (lcp_m ** 0.77) * (slope_channel_mm_safe ** -0.385)
                print(f"Tc calculado (Kirpich): {tc_min:.2f} min (L={lcp_m:.0f}m, S={slope_channel_mm_safe:.5f}m/m)")
                # Limitar Tc a un rango razonable (ej. 5 min a 10 horas)
                tc_min = max(5.0, min(tc_min, 600.0))
            else:
                print("Advertencia: No se pudo calcular Tc (faltan Lcp o Pendiente del cauce válidos). Usando valor por defecto.")
                tc_min = 60.0 # Valor por defecto si faltan datos

    except ValueError as ve:
         messagebox.showerror("Error de Datos", f"Error al obtener parámetros iniciales:\n{ve}", parent=root_window)
         # Usar valores por defecto si hay error (ej. área cero)
         area_km2 = 1.0 if np.isnan(area_km2) or area_km2 <= 0 else area_km2
         tc_min = 60.0 if np.isnan(tc_min) else tc_min
         # default_cn, default_p, default_d ya tienen valores
    except Exception as e:
        print(f"Error inesperado obteniendo parámetros automáticos: {e}")
        traceback.print_exc()
        # Usar valores por defecto si hay error inesperado
        area_km2 = 1.0 if np.isnan(area_km2) or area_km2 <= 0 else area_km2
        tc_min = 60.0 if np.isnan(tc_min) else tc_min
        # default_cn, default_p, default_d ya tienen valores

    # --- Crear la ventana del hidrograma ---
    hydrograph_window = Toplevel(root_window)
    hydrograph_window.title("Generador de Hidrograma SCS")
    hydrograph_window.geometry("650x800") # Hacerla un poco más alta para nuevas opciones
    hydrograph_window.transient(root_window)
    hydrograph_window.grab_set()

    # --- Añadir Scrollbar ---
    main_container = Frame(hydrograph_window)
    main_container.pack(fill="both", expand=True)
    vscrollbar = Scrollbar(main_container, orient="vertical")
    vscrollbar.pack(side="right", fill="y")
    canvas = tk.Canvas(main_container, yscrollcommand=vscrollbar.set, borderwidth=0, highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)
    vscrollbar.config(command=canvas.yview)
    # Frame principal dentro del canvas
    main_frame = Frame(canvas)
    canvas.create_window((0, 0), window=main_frame, anchor="nw")
    # ------------------------

    Label(main_frame, text="Generador de Hidrograma Unitario SCS",
          font=("Arial", 14, "bold")).pack(pady=(10, 15))
    Label(main_frame, text=f"Área de la Cuenca (A): {area_km2:.3f} km²",
          font=("Arial", 10, "bold")).pack(anchor="w", padx=20)

    desc_frame = Frame(main_frame, padx=20, pady=10)
    desc_frame.pack(fill="x")
    description = """
Este módulo genera un hidrograma sintético basado en el método del Hidrograma Unitario Adimensional del SCS (Soil Conservation Service, ahora NRCS).

Parámetros Clave:
- Tiempo de Concentración (Tc): Tiempo que tarda el agua en viajar desde el punto hidráulicamente más lejano hasta la salida de la cuenca. Se estima con la fórmula de Kirpich si Lcp y S_cauce están disponibles.
- Número de Curva (CN): Representa la capacidad de generación de escurrimiento de la cuenca (depende del uso/tipo de suelo y condición hidrológica).
- Precipitación (P) y Duración (D): Características de la tormenta de diseño. La duración afecta el Tiempo al Pico (Tp) y el Caudal Pico (Qp).

El Caudal Pico (Qp) se calcula usando la fórmula del SCS:
Qp = (k * A * Q) / Tp
Donde:
- Qp = Caudal Pico (m³/s)
- k = Factor de conversión de unidades (0.00278 para A en km², Q en mm, Tp en hr)
- A = Área de la cuenca (km²)
- Q = Lámina de escurrimiento (mm), calculada con P y CN.
- Tp = Tiempo al Pico (hr), relacionado con la duración (D) y Tc.

Ajuste los parámetros según su estudio hidrológico:
"""
    Label(desc_frame, text=description, wraplength=600, justify="left").pack(anchor="w")

    input_frame = Frame(main_frame, padx=20)
    input_frame.pack(fill="x", pady=5)

    def create_input_row(parent, label_text, var, default_value, unit_text):
        frame = Frame(parent)
        frame.pack(fill="x", pady=3)
        Label(frame, text=label_text, width=30, anchor="w").pack(side="left")
        # Formatear valor por defecto
        if isinstance(default_value, float):
            var.set(f"{default_value:.2f}")
        else:
            var.set(str(default_value))
        entry = Entry(frame, textvariable=var, width=12)
        entry.pack(side="left", padx=5)
        Label(frame, text=unit_text, width=10, anchor="w").pack(side="left")
        return entry

    # Variable para controlar el modo de cálculo de Tc
    tc_mode_var = tk.StringVar(value="auto")  # Valores: "auto" o "manual"

    # Variables para los inputs normales
    tc_var = tk.StringVar()
    cn_var = tk.StringVar()
    p_var = tk.StringVar()
    d_var = tk.StringVar()

    # Nueva variable para la longitud del cauce principal manual
    lcp_manual_var = tk.StringVar()
    if not np.isnan(default_lcp) and default_lcp > 0:
        lcp_manual_var.set(f"{default_lcp:.3f}")
    else:
        lcp_manual_var.set("0.000")

    # Frame para los modos de cálculo de Tc
    tc_mode_frame = Frame(input_frame)
    tc_mode_frame.pack(fill="x", pady=5)
    Label(tc_mode_frame, text="Cálculo del Tiempo de Concentración:",
          font=("Arial", 10, "bold")).pack(anchor="w")

    # Radio buttons para seleccionar el modo
    tc_auto_radio = tk.Radiobutton(tc_mode_frame, text="Automático (basado en morfometría)",
                               variable=tc_mode_var, value="auto")
    tc_auto_radio.pack(anchor="w", padx=20, pady=2)

    tc_manual_radio = tk.Radiobutton(tc_mode_frame, text="Basado en longitud del cauce principal manual",
                                 variable=tc_mode_var, value="manual")
    tc_manual_radio.pack(anchor="w", padx=20, pady=2)

    # Frame para la entrada de longitud del cauce principal manual
    lcp_frame = Frame(input_frame)
    lcp_frame.pack(fill="x", pady=3)
    Label(lcp_frame, text="Longitud del Cauce Principal:", width=30, anchor="w").pack(side="left")
    lcp_entry = Entry(lcp_frame, textvariable=lcp_manual_var, width=12)
    lcp_entry.pack(side="left", padx=5)
    Label(lcp_frame, text="km", width=10, anchor="w").pack(side="left")

    # Crear los demás campos de entrada
    tc_entry = create_input_row(input_frame, "Tiempo de Concentración (Tc):", tc_var, tc_min, "(min)")
    cn_entry = create_input_row(input_frame, "Número de Curva (CN):", cn_var, default_cn, "(1-100)")
    p_entry = create_input_row(input_frame, "Precipitación Total (P):", p_var, default_p, "(mm)")
    d_entry = create_input_row(input_frame, "Duración de Precipitación (D):", d_var, default_d, "(horas)")

    # Función para actualizar Tc cuando cambia el modo o la longitud manual
    def update_tc_calculation(*args):
        try:
            mode = tc_mode_var.get()
            if mode == "manual":
                # Usar longitud manual para calcular Tc con Kirpich
                try:
                    lcp_manual_km = float(lcp_manual_var.get())
                    if lcp_manual_km <= 0:
                        messagebox.showerror("Error", "La longitud del cauce principal debe ser mayor que cero.",
                                            parent=hydrograph_window)
                        return

                    # Usar la pendiente calculada si está disponible, o un valor por defecto
                    slope_mm_to_use = slope_channel_mm if not np.isnan(slope_channel_mm) else 0.01
                    slope_mm_to_use = max(slope_mm_to_use, 0.00001)  # Evitar divisiones por cero

                    # Calcular Tc con Kirpich
                    lcp_manual_m = lcp_manual_km * 1000.0
                    tc_calculated = 0.01947 * (lcp_manual_m ** 0.77) * (slope_mm_to_use ** -0.385)

                    # Límites razonables
                    tc_calculated = max(5.0, min(tc_calculated, 600.0))

                    # Actualizar el campo Tc
                    tc_var.set(f"{tc_calculated:.2f}")
                    print(f"Tc calculado con longitud manual: {tc_calculated:.2f} min (L={lcp_manual_m:.0f}m, S={slope_mm_to_use:.5f}m/m)")

                except ValueError:
                    messagebox.showerror("Error", "Ingrese un valor numérico válido para la longitud del cauce.",
                                        parent=hydrograph_window)
            else:
                # Modo automático - restaurar valor original calculado
                if not np.isnan(tc_min):
                    tc_var.set(f"{tc_min:.2f}")
                    print(f"Restaurando Tc calculado automáticamente: {tc_min:.2f} min")
        except Exception as e:
            print(f"Error al actualizar Tc: {e}")
            traceback.print_exc()

    # Vincular la función a cambios en el modo o la longitud
    tc_mode_var.trace_add("write", update_tc_calculation)
    lcp_manual_var.trace_add("write", update_tc_calculation)

    results_frame = Frame(main_frame, padx=20, pady=10)
    results_frame.pack(fill="x")
    Label(results_frame, text="Resultados Calculados:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5,2))
    q_label = Label(results_frame, text="Lámina Escurrimiento (Q): -- mm", anchor="w")
    q_label.pack(fill="x")
    tp_label = Label(results_frame, text="Tiempo al Pico (Tp): -- hr", anchor="w")
    tp_label.pack(fill="x")
    qp_label = Label(results_frame, text="Caudal Pico (Qp): -- m³/s", anchor="w")
    qp_label.pack(fill="x")

    note_frame = Frame(main_frame, padx=20, pady=10)
    note_frame.pack(fill="x")
    note_text = "IMPORTANTE: Los valores de Tc, CN, P y D son cruciales y deben basarse en estudios hidrológicos específicos para la cuenca y la tormenta de diseño. Los valores iniciales son solo estimaciones."
    Label(note_frame, text=note_text, wraplength=600, font=("Arial", 9, "italic"), fg="darkred", justify="left").pack(anchor="w")

    # Variables para guardar resultados intermedios y datos del hidrograma
    calculated_q = np.nan
    calculated_tp_hr = np.nan
    calculated_qp = np.nan
    hydrograph_data = None

    def calculate_intermediate_results():
        nonlocal calculated_q, calculated_tp_hr, calculated_qp
        global p
        global REL_Q_P
        try:
            tc = float(tc_var.get())
            cn_val = float(cn_var.get()) # Usar cn_val localmente aquí
            p = float(p_var.get())
            d = float(d_var.get())

            if not (1 <= cn_val <= 100): raise ValueError("CN debe estar entre 1 y 100")
            if p < 0: raise ValueError("Precipitación (P) no puede ser negativa")
            if d <= 0: raise ValueError("Duración (D) debe ser mayor que 0")
            if tc <= 0: raise ValueError("Tiempo de Concentración (Tc) debe ser mayor que 0")

            # Calcular Lámina de Escurrimiento (Q)
            s = (25400 / cn_val) - 254
            ia = 0.2 * s
            calculated_q = 0.0
            if p > ia:
                calculated_q = ((p - ia) ** 2) / (p - ia + s)
            q_label.config(text=f"Lámina Escurrimiento (Q): {calculated_q:.2f} mm")
            REL_Q_P= calculated_q/p
            print ('RELACION Q/PP = ',REL_Q_P)
            # Calcular Tiempo al Pico (Tp)
            tc_hr = tc / 60.0
            # Tiempo de retardo (lag time) Tlag = 0.6 * Tc
            tlag_hr = 0.6 * tc_hr
            # Tiempo al pico Tp = D/2 + Tlag
            calculated_tp_hr = (d / 2.0) + tlag_hr
            tp_label.config(text=f"Tiempo al Pico (Tp): {calculated_tp_hr:.3f} hr")

            # Calcular Caudal Pico (Qp)
            k_factor = 0.00278 # Factor de conversión para A(km²), Q(mm), Tp(hr) -> Qp(m³/s)
            calculated_qp = 0.0
            if calculated_tp_hr > 0:
                 calculated_qp = (k_factor * area_km2 * calculated_q) / calculated_tp_hr
                 qp_label.config(text=f"Caudal Pico (Qp): {calculated_qp:.3f} m³/s")
            else:
                 qp_label.config(text="Caudal Pico (Qp): 0.000 m³/s (Tp=0)")

            return True # Cálculo exitoso

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", str(ve), parent=hydrograph_window)
            q_label.config(text="Lámina Escurrimiento (Q): Error")
            tp_label.config(text="Tiempo al Pico (Tp): Error")
            qp_label.config(text="Caudal Pico (Qp): Error")
            calculated_q = np.nan
            calculated_tp_hr = np.nan
            calculated_qp = np.nan
            return False # Cálculo fallido
        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo intermedio:\n{e}", parent=hydrograph_window)
            q_label.config(text="Lámina Escurrimiento (Q): Error")
            tp_label.config(text="Tiempo al Pico (Tp): Error")
            qp_label.config(text="Caudal Pico (Qp): Error")
            calculated_q = np.nan
            calculated_tp_hr = np.nan
            calculated_qp = np.nan
            return False # Cálculo fallido

    def generate_and_show():
        nonlocal hydrograph_data
        # Primero, recalcular Q, Tp, Qp con los valores actuales de la interfaz
        if not calculate_intermediate_results():
            return # Salir si el cálculo intermedio falló

        # Verificar que Qp y Tp sean válidos antes de generar el hidrograma
        if np.isnan(calculated_qp) or np.isnan(calculated_tp_hr) or calculated_tp_hr <= 0:
             messagebox.showerror("Error", "No se puede generar el hidrograma sin Caudal Pico (Qp) y Tiempo al Pico (Tp) válidos y positivos.", parent=hydrograph_window)
             return

        # Tabla adimensional estándar del SCS (Tiempo/Tp vs Caudal/Qp)
        t_tp_ratios = np.array([
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
            2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0
        ])
        q_qp_ratios = np.array([
            0.000, 0.030, 0.100, 0.190, 0.310, 0.470, 0.660, 0.820, 0.930, 0.990, 1.000,
            0.990, 0.930, 0.860, 0.780, 0.680, 0.560, 0.460, 0.390, 0.330, 0.280,
            0.207, 0.147, 0.107, 0.077, 0.055, 0.029, 0.017, 0.011, 0.005
        ])

        # Calcular tiempos (en horas) y caudales (en m³/s) reales
        t_hr = t_tp_ratios * calculated_tp_hr
        q_m3s = q_qp_ratios * calculated_qp

        # Guardar datos para posible exportación
        hydrograph_data = list(zip(t_hr, q_m3s))

        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(t_hr, q_m3s, 'b-', linewidth=2, label=f'Hidrograma (Qp={calculated_qp:.2f} m³/s)')
        plt.fill_between(t_hr, q_m3s, alpha=0.3, color='blue')
        # Marcar el punto pico
        plt.plot(calculated_tp_hr, calculated_qp, 'ro', markersize=8, label=f'Pico (Tp={calculated_tp_hr:.2f} hr)')
        plt.suptitle("RELACIÓN PRECIPITACIÓN ESCURRIMIENTO = REL_Q_P")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Tiempo (horas)', fontsize=12)
        plt.ylabel('Caudal (m³/s)', fontsize=12)

        # Incluir información del modo de cálculo de Tc en el título
        tc_mode_text = "Tc manual" if tc_mode_var.get() == "manual" else "Tc auto"
        title_text = f'Hidrograma Sintético SCS\n(A={area_km2:.2f} km², CN={cn_var.get()}, P={p_var.get()}mm, D={d_var.get()}hr, {tc_mode_text}={tc_var.get()}min), REL_Q_P={REL_Q_P:.2f}'

        # Si es modo manual, incluir longitud del cauce
        if tc_mode_var.get() == "manual":
            title_text = title_text.replace(')', f', Lcp={lcp_manual_var.get()}km)')

        plt.title(title_text, fontsize=14)
        
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Habilitar botón de guardar tabla
        save_table_button.config(state=tk.NORMAL)


    def save_hydrograph_table():
        if hydrograph_data is None:
            messagebox.showwarning("Advertencia", "Primero genere el hidrograma para poder guardar la tabla.", parent=hydrograph_window)
            return

        output_path = filedialog.asksaveasfilename(
            title="Guardar Tabla del Hidrograma",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
        )
        if not output_path: return # Cancelado

        try:
            with open(output_path, 'w', newline='') as f:
                # Escribir encabezado con parámetros
                f.write(f"# Hidrograma Sintético SCS\n")
                f.write(f"# Área (A): {area_km2:.4f} km²\n")

                # Incluir modo de cálculo de Tc y parámetros relacionados
                if tc_mode_var.get() == "manual":
                    f.write(f"# Longitud Cauce Principal (manual): {lcp_manual_var.get()} km\n")
                    f.write(f"# Tiempo Concentración (Tc, calculado de Lcp manual): {tc_var.get()} min\n")
                else:
                    f.write(f"# Tiempo Concentración (Tc, automático): {tc_var.get()} min\n")

                f.write(f"# Número de Curva (CN): {cn_var.get()}\n")
                f.write(f"# Precipitación (P): {p_var.get()} mm\n")
                f.write(f"# Duración (D): {d_var.get()} hr\n")
                f.write(f"# Lámina Escurrimiento (Q): {calculated_q:.3f} mm\n")
                f.write(f"# Tiempo al Pico (Tp): {calculated_tp_hr:.4f} hr\n")
                f.write(f"# Caudal Pico (Qp): {calculated_qp:.4f} m³/s\n")
                f.write("#\n")
                # Escribir datos
                f.write("Tiempo_hr,Caudal_m3s\n")
                for t, q_val in hydrograph_data: # Renombrar q a q_val para evitar conflicto
                    f.write(f"{t:.5f},{q_val:.5f}\n")
            messagebox.showinfo("Guardado Exitoso", f"Tabla del hidrograma guardada en:\n{output_path}", parent=hydrograph_window)
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo:\n{e}", parent=hydrograph_window)


    # --- Botones de acción ---
    button_frame = Frame(main_frame, pady=15)
    button_frame.pack(fill="x")
    # Botón para calcular solo Q, Tp, Qp
    calc_button = Button(button_frame, text="Calcular Q, Tp, Qp", command=calculate_intermediate_results, width=18)
    calc_button.pack(side="left", padx=(20, 10))
    # Botón principal para generar y mostrar el gráfico
    generate_button = Button(button_frame, text="GENERAR HIDROGRAMA",
                           command=generate_and_show,
                           font=("Arial", 10, "bold"),
                           bg="#4CAF50", fg="white", width=22, height=2)
    generate_button.pack(side="left", padx=10)
    # Botón para guardar la tabla (inicialmente deshabilitado)
    save_table_button = Button(button_frame, text="Guardar Tabla (.csv)", command=save_hydrograph_table, width=18, state=tk.DISABLED)
    save_table_button.pack(side="left", padx=10)
    # Botón para cerrar
    close_button = Button(button_frame, text="Cerrar", command=hydrograph_window.destroy, width=12)
    close_button.pack(side="right", padx=(10, 20))
    

    # --- Ajustar scrollbar y calcular valores iniciales ---
    main_frame.update_idletasks() # Actualizar para obtener tamaño correcto
    canvas.config(scrollregion=canvas.bbox("all")) # Configurar región de scroll
    calculate_intermediate_results() # Calcular Q, Tp, Qp iniciales al abrir

    # --- Centrar ventana ---
    hydrograph_window.update_idletasks()
    w = hydrograph_window.winfo_width()
    h = hydrograph_window.winfo_height()
    max_h = hydrograph_window.winfo_screenheight() - 100 # Limitar altura máxima
    h = min(h, max_h)
    x = (hydrograph_window.winfo_screenwidth() // 2) - (w // 2)
    y = (hydrograph_window.winfo_screenheight() // 2) - (h // 2)
    hydrograph_window.geometry(f'{w}x{h}+{x}+{y}')
    root_window.wait_window(hydrograph_window)

def load_existing_watershed_action():
    """Acción para cargar y visualizar una cuenca existente."""
    global root_window
    root_window.withdraw()

    watershed_filepath = select_watershed_file()
    if not watershed_filepath:
        root_window.deiconify()
        return

    try:
        with rasterio.open(watershed_filepath) as src:
            watershed_data = src.read(1)
            transform = src.transform
            crs = src.crs
            extent = plotting_extent(src) # Correct: plotting_extent(src_dataset_reader)
            is_geographic = crs.is_geographic if crs else False
            src_res = src.res
            ws_nodata = src.nodata # Leer nodata del archivo de cuenca

        # Asumir valores: 0=Fuera, 1=Cuenca, 2=Río (as saved)
        # Excluir nodata si existe
        if ws_nodata is not None:
             watershed_mask = (watershed_data >= 1) & (watershed_data != ws_nodata)
             rivers_mask = (watershed_data == 2) & (watershed_data != ws_nodata)
        else:
             # If no nodata defined, assume 0 is outside
             watershed_mask = watershed_data >= 1
             rivers_mask = watershed_data == 2

        dem_np = None
        dem_transform = None
        dem_extent = extent # Initialize with watershed extent
        dem_nodata_val = None # Nodata of the associated DEM

        # Attempt to find associated DEM (same base name without suffix)
        base_dir = os.path.dirname(watershed_filepath)
        # Try to remove common suffixes like '_cuenca_drenaje_ROW_COL' or '_cuenca'
        base_name_parts = os.path.basename(watershed_filepath).split('_cuenca')[0]
        probable_dem_name = base_name_parts
        probable_dem_path = os.path.join(base_dir, probable_dem_name + ".tif") # Assume .tif extension

        print(f"Attempting to load associated DEM: {probable_dem_path}")
        if os.path.exists(probable_dem_path):
            try:
                with rasterio.open(probable_dem_path) as src_dem:
                    dem_nodata_val = src_dem.nodata # Save DEM nodata
                    # Check if DEM matches CRS, shape, and transform
                    if src_dem.crs == crs and src_dem.shape == watershed_data.shape and src_dem.transform == transform:
                        print("Associated DEM found and compatible.")
                        dem_np = src_dem.read(1)
                        dem_transform = src_dem.transform
                        dem_extent = plotting_extent(src_dem) # Correct: plotting_extent(src_dem_dataset_reader)
                    else:
                        # If not matching, try to reproject for visualization (can be slow/imprecise)
                        print(f"Warning: Associated DEM found but doesn't exactly match saved watershed (CRS/Shape/Transform). Attempting to reproject for visualization...")
                        dem_reprojected = np.zeros_like(watershed_data, dtype=src_dem.dtype)
                        # Use source DEM nodata if exists, else an unlikely value
                        src_nodata_reproj = dem_nodata_val if dem_nodata_val is not None else -9999
                        reproject(
                            source=rasterio.band(src_dem, 1),
                            destination=dem_reprojected,
                            src_transform=src_dem.transform,
                            src_crs=src_dem.crs,
                            dst_transform=transform, # Reproject to watershed transform
                            dst_crs=crs,             # Reproject to watershed CRS
                            dst_nodata=src_nodata_reproj, # Use the same nodata in destination
                            resampling=Resampling.bilinear
                        )
                        dem_np = dem_reprojected
                        dem_transform = transform # Now matches watershed
                        dem_extent = extent       # Now matches watershed (from watershed file)
                        dem_nodata_val = src_nodata_reproj # Keep the nodata used
                        print("DEM reprojected for visualization.")

            except Exception as e:
                print(f"Error loading or reprojecting associated DEM '{probable_dem_path}': {e}")
                dem_np = None # Don't use DEM if loading/reprojection fails
        else:
            print("No associated DEM found with the expected name.")

        # Calculate area of the loaded watershed
        area_km2 = calculate_area_km2(watershed_mask, transform, is_geographic, src_res=src_res)

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))

        # Show DEM in background if loaded
        if dem_np is not None:
            # Mask DEM nodata for visualization
            dem_plot = np.ma.masked_equal(dem_np, dem_nodata_val) if dem_nodata_val is not None else dem_np
            # Corrected: plotting_extent should be called with (width, height) and transform
            # dem_np.shape is (rows, cols), so (cols, rows) is dem_np.shape[::-1]
            dem_plot_extent = plotting_extent(dem_np, dem_transform) # CORRECCIÓN AQUÍ
            show(dem_plot, ax=ax, cmap='terrain', transform=dem_transform, extent=dem_plot_extent, alpha=0.7)
            # Update the main extent to be the union of watershed and DEM if DEM was loaded
            # This is a simplification, a proper union would be more complex
            extent = dem_plot_extent # Use the DEM's extent if it was successfully loaded and plotted

        # Show Rivers inside the watershed (if they exist)
        if np.any(rivers_mask):
            river_color = np.zeros((*rivers_mask.shape, 4), dtype=np.float32)
            river_color[rivers_mask] = [0, 0, 0.8, 0.9] # Dark blue more opaque
            # Use 'nearest' to avoid blurring rivers
            ax.imshow(river_color, extent=extent, interpolation='nearest', zorder=2)

        # Draw watershed boundary
        try:
            shapes_gen = shapes(watershed_mask.astype(rasterio.uint8), mask=watershed_mask, transform=transform)
            basin_shape = None
            all_shapes = [shape(geom) for geom, value in shapes_gen if value == 1]
            
            if all_shapes:
                if len(all_shapes) > 1:
                    basin_shape = MultiPolygon(all_shapes)
                else:
                    basin_shape = all_shapes[0]

                if basin_shape.geom_type == 'Polygon':
                    x_poly, y_poly = basin_shape.exterior.xy
                    ax.plot(x_poly, y_poly, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)
                    for interior in basin_shape.interiors:
                        x_hole, y_hole = interior.xy
                        ax.plot(x_hole, y_hole, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)
                elif basin_shape.geom_type == 'MultiPolygon':
                     for poly in basin_shape.geoms:
                         if poly.geom_type == 'Polygon':
                             x_poly, y_poly = poly.exterior.xy
                             ax.plot(x_poly, y_poly, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)
                             for interior in poly.interiors:
                                 x_hole, y_hole = interior.xy
                                 ax.plot(x_hole, y_hole, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)

        except Exception as shape_err:
            print(f"Error generating boundary from shapes: {shape_err}. Using contour.")
            # Fallback to matplotlib.contour if shapes fails
            x_coords = np.linspace(extent[0], extent[1], watershed_mask.shape[1])
            y_coords = np.linspace(extent[2], extent[3], watershed_mask.shape[0])
            X, Y = np.meshgrid(x_coords, y_coords)
            ax.contour(X, Y, watershed_mask.astype(int), levels=[0.5], colors='black', linewidths=1.5, zorder=4)


        # Add text with area
        ax.text(0.02, 0.98, f"Área: {area_km2:.3f} km²",
                transform=ax.transAxes, size=12, ha="left", va="top",
                bbox=dict(boxstyle="round", fc="white", ec="blue", alpha=0.8))

        ax.set_title(f"Cuenca Cargada: {os.path.basename(watershed_filepath)}")
        ax.set_xlabel("Coordenada X / Longitud")
        ax.set_ylabel("Coordenada Y / Latitud")
        ax.ticklabel_format(style='plain', axis='both', useOffset=False)
        # Adjust limits to watershed extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        plt.tight_layout()
        plt.show()

    except rasterio.RasterioIOError as rio_err:
         messagebox.showerror("Error de Archivo", f"No se pudo leer el archivo de cuenca:\n{watershed_filepath}\nVerifique que el archivo exista, no esté corrupto y tenga permisos de lectura.\nError: {rio_err}")
         print(f"Error al leer cuenca: {rio_err}")
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar o visualizar la cuenca:\n{e}")
        print(f"Error al cargar cuenca: {e}")
        traceback.print_exc()
    finally:
        # Ensure main window reappears
        root_window.deiconify()

# --- Función Intermedia para Coordenadas Manuales ---
def manual_coordinate_workflow(threshold):
    """Gestiona la obtención de coordenadas manuales y llama a main_action."""
    global root_window
    coords = ask_manual_coordinates(root_window)
    if coords[0] is not None and coords[1] is not None:
        main_action(threshold, use_manual_coords=True, manual_coords=coords)
    else:
        print("Entrada de coordenadas cancelada o inválida.")

# --- Función Principal de Delineación ---

def main_action(accumulation_threshold, use_manual_coords=False, manual_coords=None):
    """Función principal que ejecuta el proceso de delineación."""
    global clicked_coords, root_window, last_watershed_data, morphometry_button, runoff_button, hydrograph_button, SIMULADOR, STRAHLER_MAP

    # --- Resetear estado ---
    if morphometry_button: morphometry_button.config(state=tk.DISABLED)
    if runoff_button: runoff_button.config(state=tk.DISABLED)
    if hydrograph_button: hydrograph_button.config(state=tk.DISABLED)
    if STRAHLER_MAP: STRAHLER_MAP.config(state=tk.DISABLED) # Disable new Strahler map button
    last_watershed_data["calculated"] = False
    for key in last_watershed_data:
        if key != "calculated": last_watershed_data[key] = None
    if not use_manual_coords:
        clicked_coords = [None, None]
    
    # -----------------------

    root_window.withdraw() # Ocultar ventana principal durante procesamiento

    # 1. Seleccionar DEM
    dem_filepath = select_dem_file()
    if not dem_filepath:
        print("Selección de DEM cancelada.")
        root_window.deiconify() # Mostrar ventana principal si se cancela
        return

    pysheds_grid = None # Inicializar pysheds_grid a None

    try:
        # 2. Cargar DEM y Metadatos con Rasterio (para info y visualización inicial)
        print(f"Cargando DEM: {dem_filepath}")
        dem_nodata = None # Inicializar
        with rasterio.open(dem_filepath) as src:
            transform = src.transform
            crs = src.crs
            extent = plotting_extent(src) # Correct: plotting_extent(src_dataset_reader)
            is_geographic = crs.is_geographic if crs else False
            dem_shape = src.shape
            dem_bounds = src.bounds # Límites geográficos/proyectados
            src_res = src.res # Resolución (dx, dy)
            dem_nodata = src.nodata # Leer nodata original del archivo

        print(f"DEM cargado: {dem_shape[0]}x{dem_shape[1]} píxeles.")
        print(f"CRS: {crs}")
        print(f"Extent (izq, der, abajo, arriba): {extent}")
        print(f"Resolución (x, y): {src_res}")
        if dem_nodata is not None:
            print(f"Valor Nodata detectado en el archivo: {dem_nodata}")
        else:
            print("No se detectó un valor Nodata específico en el archivo (podría usar NaN internamente).")

        if is_geographic:
            print("Tipo CRS: Geográfico (unidades: grados)")
        else:
            print("Tipo CRS: Proyectado (unidades: verificar CRS, asumidas metros)")
            if crs and hasattr(crs, 'linear_units') and crs.linear_units and crs.linear_units.lower() != 'metre':
                 messagebox.showwarning("Advertencia de Unidades",
                                       f"El CRS proyectado '{crs.name}' parece usar unidades '{crs.linear_units}'.\n"
                                       "El programa asume METROS para los cálculos de distancia/área.\n"
                                       "Los resultados pueden ser incorrectos si las unidades no son metros.")

        # 3. Procesamiento Hidrológico con PySheds
        print("Inicializando PySheds Grid...")
        pysheds_grid = Grid.from_raster(dem_filepath)

        print("Leyendo DEM con PySheds...")
        dem_pysheds = pysheds_grid.read_raster(dem_filepath)

        dem_np = pysheds_grid.view(dem_pysheds)
        original_nodata = dem_nodata
        nan_count = np.sum(np.isnan(dem_np))
        nodata_count = 0
        if original_nodata is not None:
            if isinstance(original_nodata, float):
                nodata_count = np.sum(np.isclose(dem_np, original_nodata))
            else:
                nodata_count = np.sum(dem_np == original_nodata)

        print(f"Encontrados {nan_count} valores NaN y {nodata_count} valores nodata ({original_nodata})")

        if nan_count > 0 or (nodata_count > 0 and original_nodata is not None):
            print("Modificando valores NaN/nodata en el DEM...")

            dem_name = None
            for attr_name in dir(pysheds_grid):
                attr = getattr(pysheds_grid, attr_name)
                if isinstance(attr, np.ndarray) and attr.shape == dem_np.shape:
                    if np.array_equal(attr, dem_np, equal_nan=True):
                        dem_name = attr_name
                        break

            if dem_name:
                print(f"Found DEM attribute: {dem_name}")
                dem_array = getattr(pysheds_grid, dem_name)

                if nan_count > 0:
                    print(f"Replacing {nan_count} NaN values with -9999...")
                    dem_array[np.isnan(dem_array)] = -9999

                if nodata_count > 0 and not np.isnan(original_nodata):
                    print(f"Replacing {nodata_count} nodata values ({original_nodata}) with -9999...")
                    if isinstance(original_nodata, float):
                        dem_array[np.isclose(dem_array, float(original_nodata))] = -9999
                    else:
                        dem_array[dem_array == original_nodata] = -9999

                print("NaN and nodata values modified in the DEM.")
            else:
                print("Could not find the DEM array within the Grid object. Attempting to process without modification.")

        dem_check_np = pysheds_grid.view(dem_pysheds)
        invalid_mask = (dem_check_np == -9999)
        if dem_nodata is not None and not np.isnan(dem_nodata) and dem_nodata != -9999:
            try:
                if isinstance(dem_nodata, float) or dem_check_np.dtype.kind == 'f':
                    invalid_mask |= np.isclose(dem_check_np, float(dem_nodata))
                else:
                    invalid_mask |= (dem_check_np == dem_nodata)
            except (TypeError, ValueError): pass
        valid_mask = ~invalid_mask
        num_valid = np.sum(valid_mask)
        num_total_check = dem_check_np.size
        valid_percentage = (100 * num_valid / num_total_check) if num_total_check > 0 else 0
        print(f"DEM check (PySheds): {num_valid}/{num_total_check} ({valid_percentage:.2f}%) valid cells (considering -9999 and original nodata).")

        if num_valid == 0:
            messagebox.showerror("Data Error", "The DEM file contains no valid data cells after being read by PySheds.\nIt appears to be empty or contain only Nodata/NaN values.")
            root_window.deiconify()
            return
        elif valid_percentage < 50:
             messagebox.showwarning("Data Warning", f"The DEM file contains a large number of Nodata/NaN cells ({100-valid_percentage:.2f}%).\nHydrological processing might fail or give unexpected results.")

        if num_valid > 0:
             min_elev = np.min(dem_check_np[valid_mask])
             max_elev = np.max(dem_check_np[valid_mask])
             print(f"Valid elevation range in DEM: {min_elev:.2f} - {max_elev:.2f}")

        print("Processing DEM (fill depressions, resolve flats)...")
        try:
            print("Attempting to reduce processing size...")
            dem_valid = dem_check_np != -9999
            if np.any(dem_valid):
                rows_valid, cols_valid = np.where(dem_valid)
                min_row, max_row = np.min(rows_valid), np.max(rows_valid)
                min_col, max_col = np.min(cols_valid), np.max(cols_valid)

                margin = 10
                min_row = max(0, min_row - margin)
                max_row = min(dem_check_np.shape[0], max_row + margin)
                min_col = max(0, min_col - margin)
                max_col = min(dem_check_np.shape[1], max_col + margin)

                print(f"Processing subregion: rows {min_row}-{max_row}, columns {min_col}-{max_col}")

            filled_dem = pysheds_grid.fill_depressions(dem_pysheds)
            inflated_dem = pysheds_grid.resolve_flats(filled_dem)
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            print("Attempting with alternative options for large files...")

            try:
                try:
                    filled_dem = pysheds_grid.fill_depressions(dem_pysheds, inplace=True)
                    inflated_dem = pysheds_grid.resolve_flats(filled_dem if filled_dem is not None else dem_pysheds, inplace=True)
                    if inflated_dem is None:
                        inflated_dem = dem_pysheds
                except TypeError:
                    filled_dem = pysheds_grid.fill_depressions(dem_pysheds)
                    inflated_dem = pysheds_grid.resolve_flats(filled_dem)
            except Exception as e2:
                try:
                    print("Attempting to skip DEM preprocessing...")
                    filled_dem = dem_pysheds  # Use the original DEM as fallback
                    inflated_dem = dem_pysheds
                except Exception as e3:
                    error_msg = f"Error in hydrological processing: {e3}\n\n"
                    error_msg += "Suggestions:\n"
                    error_msg += "1. Try with a smaller DEM (clip your area of interest)\n"
                    error_msg += "2. Verifique que el DEM no tenga demasiadas celdas nodata\n"
                    error_msg += "3. Asegúrese que el DEM tenga un valor nodata válido"
                    messagebox.showerror("Error de Procesamiento Hidrológico", error_msg)
                    print(f"Error in fill_depressions/resolve_flats: {e3}")
                    traceback.print_exc()
                    root_window.deiconify()
                    return

        print("Calculating flow direction (D8)...")
        fdir = pysheds_grid.flowdir(inflated_dem, routing='d8')
        pysheds_grid.fdir = fdir

        print("Calculating flow accumulation...")
        facc = pysheds_grid.accumulation(fdir, routing='d8')
        print("Flow direction and accumulation calculated.")

        # NUEVO: Calcular orden de Strahler
        print("Calculando orden de Strahler...")
        strahler_order = pysheds_grid.stream_order(fdir, facc, routing='d8')
        strahler_order_np = pysheds_grid.view(strahler_order).astype(float)
        print(f"Orden de Strahler máximo calculado: {np.nanmax(strahler_order_np)}")        
        
        fdir_np = pysheds_grid.view(fdir)
        facc_np = pysheds_grid.view(facc)
        dem_display_np = pysheds_grid.view(dem_pysheds)
        inflated_dem_np = pysheds_grid.view(inflated_dem)

        print("Checking flow accumulation results...")
        facc_valid = ~np.isnan(facc_np) & np.isfinite(facc_np)
        valid_facc_count = np.sum(facc_valid)
        valid_facc_percent = (valid_facc_count / facc_np.size) * 100
        print(f"Flow accumulation: {valid_facc_count} valid cells ({valid_facc_percent:.2f}%)")

        if valid_facc_count == 0:
            messagebox.showerror("Error", "No valid flow accumulation values found. The DEM might have issues or be too large.")
            root_window.deiconify()
            return

        min_valid_facc = np.min(facc_np[facc_valid])
        max_valid_facc = np.max(facc_np[facc_valid])
        mean_valid_facc = np.mean(facc_np[facc_valid])
        median_valid_facc = np.median(facc_np[facc_valid])
        std_valid_facc = np.std(facc_np[facc_valid])

        print(f"Accumulation stats: Min={min_valid_facc}, Max={max_valid_facc}, Mean={mean_valid_facc:.2f}, Median={median_valid_facc}, Std={std_valid_facc:.2f}")

        if accumulation_threshold > max_valid_facc:
            adjusted_threshold = max(1, max_valid_facc * 0.1)
            print(f"Threshold too high! Automatically adjusted from {accumulation_threshold} to {adjusted_threshold:.2f}")
            accumulation_threshold = adjusted_threshold

        # 4. Extract Drainage Network based on Threshold
        print(f"Extracting drainage network with umbral = {accumulation_threshold} celdas")
        # This is the ORIGINAL network, used for Strahler and as base for enhancement
        original_rivers_raster_np = (facc_np > accumulation_threshold) & facc_valid
        num_original_river_cells = np.sum(original_rivers_raster_np)
        num_total = dem_shape[0] * dem_shape[1]
        print(f"Original drainage network extracted: {num_original_river_cells} cells ({100 * num_original_river_cells / num_total:.2f}%)")

        if num_original_river_cells == 0:
            reduced_threshold = max(1.0, accumulation_threshold / 10)
            print(f"No se detectaron ríos. Reduciendo umbral a {reduced_threshold}")
            original_rivers_raster_np = (facc_np > reduced_threshold) & facc_valid
            num_original_river_cells = np.sum(original_rivers_raster_np)
            print(f"Red de drenaje con umbral reducido: {num_original_river_cells} celdas ({100 * num_original_river_cells / num_total:.2f}%)")
            accumulation_threshold = reduced_threshold

        if num_original_river_cells == 0:
            messagebox.showerror("Error", "No se pudo extraer la red de drenaje. Intente con un umbral más bajo o revise el DEM.")
            root_window.deiconify()
            return

        # Enhance the drainage network for better continuity (for display and snapping)
        print("Mejorando la continuidad de la red de drenaje para visualización y snapping...")
        rivers_for_display_and_snap = enhance_drainage_network(original_rivers_raster_np, iterations=2)
        num_enhanced_river_cells = np.sum(rivers_for_display_and_snap)
        print(f"Red de drenaje mejorada: {num_enhanced_river_cells} celdas ({100 * num_enhanced_river_cells / num_total:.2f}%)")


        # 5. Get Outlet Point
        outlet_x, outlet_y = None, None
        if use_manual_coords:
            if manual_coords and manual_coords[0] is not None and manual_coords[1] is not None:
                outlet_x, outlet_y = manual_coords
                print(f"Using manual coordinates: X={outlet_x:.4f}, Y={outlet_y:.4f}")
                if not (dem_bounds.left <= outlet_x <= dem_bounds.right and
                        dem_bounds.bottom <= outlet_y <= dem_bounds.top):
                     messagebox.showerror("Coordinate Error",
                                          f"Manual coordinates ({outlet_x:.4f}, {outlet_y:.4f}) "
                                          f"are outside the DEM bounds:\n"
                                          f"X: [{dem_bounds.left:.4f}, {dem_bounds.right:.4f}]\n"
                                          f"Y: [{dem_bounds.bottom:.4f}, {dem_bounds.top:.4f}]")
                     root_window.deiconify()
                     return
            else:
                messagebox.showerror("Internal Error", "Manual coordinates expected but not provided.")
                root_window.deiconify()
                return
        else:
            print("Preparando visualización para selección de punto de salida...")
            fig_select, ax_select = plt.subplots(figsize=(10, 8))

            dem_plot_select = dem_display_np.copy()
            mask_select = np.isnan(dem_plot_select) | (dem_plot_select == -9999)
            if dem_nodata is not None and not np.isnan(dem_nodata) and dem_nodata != -9999:
                 try:
                     if isinstance(dem_nodata, float) or dem_plot_select.dtype.kind == 'f':
                         mask_select |= np.isclose(dem_plot_select, float(dem_nodata))
                     else:
                         mask_select |= (dem_plot_select == dem_nodata)
                 except (TypeError, ValueError): pass
            dem_plot_select_masked = np.ma.masked_where(mask_select, dem_plot_select)

            show(dem_plot_select_masked, ax=ax_select, cmap='terrain', transform=transform, extent=extent, alpha=0.7)

            print("Preparando red de drenaje para visualización...")
            river_color_display = np.zeros((*rivers_for_display_and_snap.shape, 4))
            river_color_display[rivers_for_display_and_snap] = [0, 0, 1, 0.95]
            ax_select.imshow(river_color_display, extent=extent, interpolation='nearest', zorder=2)

            print(f"Visualizando red de drenaje mejorada: {np.sum(rivers_for_display_and_snap)} celdas de río mostradas")

            cid = fig_select.canvas.mpl_connect('button_press_event', onclick)

            ax_select.set_title("Seleccione el Punto de Salida\n"
                                "*** HAGA CLIC SOBRE UN CAUCE (LÍNEA AZUL) ***\n"
                                "(Use las herramientas Lupa/Flechas de la barra inferior para Zoom/Pan ANTES de hacer clic)",
                                fontsize=12)
            ax_select.set_xlabel("Coordenada X / Longitud")
            ax_select.set_ylabel("Coordenada Y / Latitud")
            ax_select.ticklabel_format(style='plain', axis='both', useOffset=False)
            ax_select.set_xlim(extent[0], extent[1])
            ax_select.set_ylim(extent[2], extent[3])

            print("\n*********************************************************************")
            print("Mostrando mapa interactivo.")
            print("IMPORTANTE: Use las herramientas de Zoom (lupa) y Pan (flechas)")
            print("en la barra de herramientas del gráfico para acercarse ANTES de hacer clic.")
            print("Asegúrese de DESACTIVAR el modo Zoom/Pan antes de seleccionar el punto.")
            print("Haga clic en el punto de salida deseado sobre la red azul.")
            print("*********************************************************************\n")

            plt.show()
            fig_select.canvas.mpl_disconnect(cid)

            if clicked_coords[0] is None or clicked_coords[1] is None:
                print("No point selected on the map. Aborting.")
                root_window.deiconify()
                return
            outlet_x, outlet_y = clicked_coords

        # 6. Snap Outlet Point to River
        print(f"\nSnapping point ({outlet_x:.4f}, {outlet_y:.4f}) to nearest river cell with high accumulation...")
        # Use the enhanced network for snapping, as it's what the user sees
        snapped_coords_rc = snap_to_river(facc_np, rivers_for_display_and_snap, (outlet_x, outlet_y), transform, search_radius=30)

        if snapped_coords_rc is None:
            print("Snap to river failed.")
            root_window.deiconify()
            return

        outlet_row, outlet_col = int(snapped_coords_rc[0]), int(snapped_coords_rc[1])
        snapped_x, snapped_y = rasterio.transform.xy(transform, outlet_row, outlet_col)
        print(f"Point snapped to: Row={outlet_row}, Column={outlet_col} (X={snapped_x:.4f}, Y={snapped_y:.4f})")

        test_watershed = delineate_watershed_robust(fdir_np, outlet_row, outlet_col, max_cells=10000)
        test_size = np.sum(test_watershed)
        print(f"Estimated watershed size with snapped point: {test_size} cells")

        if test_size < 500:
            print("Snapped point would generate a very small watershed. Searching for better location...")
            main_channel_rc = find_main_channel_point(facc_np, fdir_np, rivers_for_display_and_snap, (outlet_x, outlet_y), transform, search_radius=100)
    
            if main_channel_rc and main_channel_rc[0] is not None:
                alt_row, alt_col = main_channel_rc
                alt_x, alt_y = rasterio.transform.xy(transform, alt_row, alt_col)
                message = f"Se detectó que el punto seleccionado generaría una cuenca muy pequeña ({test_size} celdas).\n\n"
                message += f"¿Desea usar un punto alternativo que generaría una cuenca más grande?"
        
                if messagebox.askyesno("Punto Alternativo Recomendado", message):
                    outlet_row, outlet_col = alt_row, alt_col
                    snapped_x, snapped_y = alt_x, alt_y
                    print(f"Usando punto alternativo: Fila={outlet_row}, Columna={alt_col} (X={snapped_x:.4f}, Y={snapped_y:.4f})")
                else:
                    print("Usuario eligió mantener el punto original a pesar del tamaño pequeño")

        if not (0 <= outlet_row < dem_shape[0] and 0 <= outlet_col < dem_shape[1]):
            messagebox.showerror("Internal Error", f"El punto ajustado ({outlet_row}, {outlet_col}) está fuera de los límites del DEM después del ajuste.")
            root_window.deiconify()
            return

        # 7. Delineate the Watershed using the robust function
        print("Delineating watershed from the snapped point...")
        watershed_np = delineate_watershed_robust(fdir_np, outlet_row, outlet_col)
        num_watershed_cells = np.sum(watershed_np)

        print(f"Checking delineated watershed: {num_watershed_cells} cells")
        if num_watershed_cells <= 1:
            messagebox.showerror("Delineation Error", "La delineación de la cuenca falló o resultó en un área muy pequeña (<=1 celda).\nVerifique el punto de salida, el DEM y la dirección de flujo calculada.")
            root_window.deiconify()
            return
        print(f"Cuenca delineada con {num_watershed_cells} celdas.")


        # 8. Calculate Area and Prepare Final Data
        area_km2 = calculate_area_km2(watershed_np, transform, is_geographic, src_res)
        
        # Mask of rivers INSIDE the delineated watershed (for display)
        rivers_in_watershed_display_mask = np.logical_and(rivers_for_display_and_snap, watershed_np)
        # Mask of original rivers INSIDE the delineated watershed (for Strahler calculation)
        rivers_in_watershed_for_strahler_mask = np.logical_and(original_rivers_raster_np, watershed_np)


        # --- Save data for later calculations ---
        last_watershed_data["watershed_mask"] = watershed_np.copy()
        last_watershed_data["rivers_mask"] = rivers_in_watershed_display_mask.copy() # Enhanced rivers within watershed
        last_watershed_data["rivers_for_strahler_mask"] = rivers_in_watershed_for_strahler_mask.copy() # Original rivers within watershed
        last_watershed_data["dem_data"] = dem_display_np # Original DEM read by pysheds
        last_watershed_data["inflated_dem"] = inflated_dem_np # Processed DEM
        last_watershed_data["transform"] = transform
        last_watershed_data["src_res"] = src_res
        last_watershed_data["is_geographic"] = is_geographic
        last_watershed_data["pysheds_grid"] = pysheds_grid # Save the renamed Grid object
        last_watershed_data["outlet_coords"] = (outlet_row, outlet_col) # Save snapped coords (row, col)
        last_watershed_data["fdir"] = fdir_np # Save the numpy array of flow direction
        last_watershed_data["strahler_order_data"] = strahler_order_np.copy() # NUEVO: Guardar orden de Strahler
        last_watershed_data["calculated"] = True

        # Enable buttons for subsequent calculations
        if morphometry_button: morphometry_button.config(state=tk.NORMAL)
        if runoff_button: runoff_button.config(state=tk.NORMAL)
        if hydrograph_button: hydrograph_button.config(state=tk.NORMAL)
        if SIMULADOR: SIMULADOR.config(state=tk.NORMAL)
        if STRAHLER_MAP: STRAHLER_MAP.config(state=tk.NORMAL) # Enable new Strahler map button
        # -----------------------------------------------------------------

        # 9. Visualización Final
        print("Preparando visualización final...")
        fig_final, ax_final = plt.subplots(figsize=(10, 8))

        dem_plot_final = dem_display_np.copy()
        mask_final = np.isnan(dem_plot_final) | (dem_plot_final == -9999)
        if dem_nodata is not None and not np.isnan(dem_nodata) and dem_nodata != -9999:
             try:
                 if isinstance(dem_nodata, float) or dem_plot_final.dtype.kind == 'f':
                     mask_final |= np.isclose(dem_plot_final, float(dem_nodata))
                 else:
                     mask_final |= (dem_plot_final == dem_nodata)
             except (TypeError, ValueError): pass
        dem_plot_final_masked = np.ma.masked_where(mask_final, dem_plot_final)

        show(dem_plot_final_masked, ax=ax_final, cmap='terrain', transform=transform, extent=extent, alpha=0.7)

        # NUEVO: Plotear ríos por orden de Strahler con líneas más delgadas
        print("Preparando ríos por orden de Strahler para visualización final...")
        strahler_rivers_in_ws = last_watershed_data["strahler_order_data"].copy()
        strahler_rivers_in_ws[~rivers_in_watershed_for_strahler_mask] = np.nan # Usar la máscara original

        max_order = int(np.nanmax(strahler_rivers_in_ws)) if not np.isnan(np.nanmax(strahler_rivers_in_ws)) else 1
        if max_order == 0: max_order = 1

        # Usar un mapa de colores que tenga buen contraste entre órdenes consecutivos
        cmap = plt.cm.tab10  # Cambiado a tab10 para mejor distinción visual
        norm = plt.Normalize(vmin=1, vmax=max_order)

        # Dibujar los ríos en orden inverso (del más alto al más bajo) para que los de orden superior
        # queden encima y sean más visibles
        for order in range(max_order, 0, -1):
            order_mask = (strahler_rivers_in_ws == order)
            if np.any(order_mask):
                # Reducir o eliminar la dilatación para órdenes bajos y mantener líneas delgadas
                # Solo aplicar dilatación mínima para órdenes superiores para hacerlos visibles
                if order >= 3:  # Órdenes altos: dilatación mínima
                    dilated_order_mask = binary_dilation(order_mask, iterations=1)
                else:  # Órdenes bajos: sin dilatación o muy poca
                    dilated_order_mask = order_mask if order == 1 else binary_dilation(order_mask, iterations=1)

                color = cmap(norm(order))
                # Aumentar la opacidad con el orden para que los órdenes superiores destaquen más
                alpha = min(0.7 + order * 0.05, 0.95)

                order_color_array = np.zeros((*dem_shape, 4), dtype=np.float32)
                order_color_array[dilated_order_mask] = [*color[:3], alpha]

                ax_final.imshow(order_color_array, extent=extent, interpolation='nearest',
                                zorder=2 + order)  # Orden más alto encima

        # Añadir una barra de color para el orden de Strahler
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig_final.colorbar(sm, ax=ax_final, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Orden de Strahler')
        cbar.set_ticks(np.arange(1, max_order + 1))
        cbar.set_ticklabels([str(i) for i in np.arange(1, max_order + 1)])


        # Dibujar contorno de la cuenca (using shapes if possible)
        try:
            shapes_gen_final = shapes(watershed_np.astype(rasterio.uint8), mask=watershed_np, transform=transform)
            all_shapes_final = [shape(geom) for geom, value in shapes_gen_final if value == 1]

            if all_shapes_final:
                if len(all_shapes_final) > 1:
                    basin_shape_final = MultiPolygon(all_shapes_final)
                else:
                    basin_shape_final = all_shapes_final[0]

                if basin_shape_final.geom_type == 'Polygon':
                    x_poly, y_poly = basin_shape_final.exterior.xy
                    ax_final.plot(x_poly, y_poly, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)
                    for interior in basin_shape_final.interiors:
                        x_hole, y_hole = interior.xy
                        ax_final.plot(x_hole, y_hole, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)
                elif basin_shape_final.geom_type == 'MultiPolygon':
                     for poly in basin_shape_final.geoms:
                         if poly.geom_type == 'Polygon':
                             x_poly, y_poly = poly.exterior.xy
                             ax_final.plot(x_poly, y_poly, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)
                             for interior in poly.interiors:
                                 x_hole, y_hole = interior.xy
                                 ax_final.plot(x_hole, y_hole, color='black', linewidth=1.5, solid_capstyle='round', zorder=4)

        except Exception as shape_err_final:
            print(f"Error generating final boundary from shapes: {shape_err_final}. Using contour.")
            x_coords_final = np.linspace(extent[0], extent[1], dem_shape[1])
            y_coords_final = np.linspace(extent[2], extent[3], dem_shape[0])
            X_final, Y_final = np.meshgrid(x_coords_final, y_coords_final)
            ax_final.contour(X_final, Y_final, watershed_np.astype(int), levels=[0.5], colors='black', linewidths=1.5, zorder=4)


        # Marcar el punto de salida ajustado
        ax_final.plot(snapped_x, snapped_y, 'ro', markersize=5, markeredgecolor='black',
                      label=f'Aforo a la parcela)', zorder=5)
        ax_final.set_title(f"Microcuenca que drena a la parcela o punto de interés")
        ax_final.set_xlabel("Coordenada X / Longitud")
        ax_final.set_ylabel("Coordenada Y / Latitud")
        ax_final.ticklabel_format(style='plain', axis='both', useOffset=False)
        # Add area to plot
        ax_final.text(0.02, 0.98, f"Área: {area_km2:.3f} km²",
                     transform=ax_final.transAxes, size=12, ha="left", va="top",
                     bbox=dict(boxstyle="round", fc="white", ec="blue", alpha=0.8))
        ax_final.set_xlim(extent[0], extent[1])
        ax_final.set_ylim(extent[2], extent[3])
        ax_final.legend(loc='lower right')

        print("Showing final map...")
        plt.tight_layout()
        plt.show()

        # 10. Save Option
        save_choice = messagebox.askyesno("Save Result", "¿Desea guardar el mapa de la cuenca y red de drenaje como un archivo GeoTIFF?")
        if save_choice:
            # Suggest filename based on DEM and outlet coordinates
            base_dir = os.path.dirname(dem_filepath)
            base_name = os.path.splitext(os.path.basename(dem_filepath))[0]
            output_path_suggestion = os.path.join(base_dir, f"{base_name}_cuenca_drenaje_{outlet_row}_{outlet_col}.tif")
            output_path = filedialog.asksaveasfilename(
                title="Guardar GeoTIFF (Cuenca y Drenaje)",
                initialfile=output_path_suggestion,
                defaultextension=".tif",
                filetypes=[("GeoTIFF files", "*.tif *.tiff")]
            )
            if output_path:
                try:
                    # Create output raster: 0=Fuera, 1=Cuenca, 2=Río (sin orden de Strahler)
                    output_raster = np.zeros(dem_shape, dtype=rasterio.uint8)
                    output_raster[watershed_np] = 1 # Marcar área de la cuenca
                    output_raster[rivers_in_watershed_display_mask] = 2 # Sobrescribir ríos dentro de la cuenca (enhanced for display)
                    # Usar perfil del DEM original pero ajustar dtype y nodata
                    profile = rasterio.open(dem_filepath).profile
                    profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress='lzw') # Usar 0 como nodata para la salida
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(output_raster, 1)
                        # Opcional: Añadir descripción a las etiquetas
                        dst.update_tags(1, DESCRIPTION="Mapa de Cuenca y Drenaje: 0=Fuera, 1=Cuenca, 2=Río")
                    print(f"Cuenca y red de drenaje guardadas en: {output_path}")
                    messagebox.showinfo("Guardado Exitoso", f"Cuenca y red de drenaje guardadas en:\n{output_path}\nValores: 0=Fuera, 1=Cuenca, 2=Río")
                except Exception as e_save:
                    messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo GeoTIFF:\n{e_save}")
                    print(f"Error saving file: {e_save}")
            else:
                print("Guardado cancelado por el usuario.")

    except rasterio.RasterioIOError as rio_err:
         messagebox.showerror("Error de Archivo", f"No se pudo leer el archivo DEM:\n{dem_filepath}\nVerifique que el archivo exista, no esté corrupto y tenga permisos de lectura.\nError: {rio_err}")
         print(f"Error al leer DEM: {rio_err}")
         last_watershed_data["calculated"] = False # Marcar como no calculado
    except ImportError as imp_err:
         messagebox.showerror("Error de Dependencia", f"Falta una librería requerida: {imp_err}\nAsegúrese de tener instalados pysheds, rasterio, shapely, matplotlib, numpy, scipy.")
         print(f"Error de importación: {imp_err}")
         last_watershed_data["calculated"] = False
    # Catch specifically the TypeError from nodata/dtype that PySheds might throw
    except TypeError as type_err:
         if 'nodata` value not representable in dtype' in str(type_err):
              messagebox.showerror("Error de Tipo de Datos",
                                   f"El valor 'nodata' ({dem_nodata}) definido en el archivo DEM no es compatible con el tipo de datos del DEM.\n"
                                   "PySheds no pudo procesarlo.\n"
                                   "Intente corregir el valor nodata en el archivo GeoTIFF usando un SIG (ej. QGIS, ArcGIS) o deje que PySheds lo maneje (como se intentó).")
              print(f"Error de nodata/dtype: {type_err}")
         else: # Other unexpected TypeError
              messagebox.showerror("Error de Tipo", f"Ocurrió un error de tipo inesperado:\n{type_err}")
              print(f"TypeError general: {type_err}")
         traceback.print_exc()
         last_watershed_data["calculated"] = False
    except Exception as e:
        # General catch for other unexpected errors
        messagebox.showerror("Error de Procesamiento", f"Ocurrió un error inesperado durante la delineación:\n{e}")
        print(f"Error general en main_action: {e}")
        traceback.print_exc() # Imprimir detalles completos del error en la consola
        last_watershed_data["calculated"] = False # Marcar como no calculado
        # Asegurar que los botones estén deshabilitados si falla
        if morphometry_button: morphometry_button.config(state=tk.DISABLED)
        if runoff_button: runoff_button.config(state=tk.DISABLED)
        if hydrograph_button: hydrograph_button.config(state=tk.DISABLED)
        if STRAHLER_MAP: STRAHLER_MAP.config(state=tk.DISABLED) # Disable new Strahler map button
    finally:
        # Asegurar que la ventana principal sea visible al final, incluso si hubo un error
        if 'root_window' in globals() and root_window:
            root_window.deiconify() # Mostrar ventana principal
            root_window.lift() # Traer al frente
            root_window.focus_force() # Forzar el foco

# Función para mostrar un mapa detallado del orden de Strahler
def show_strahler_map():
    """Muestra un mapa detallado de los ríos con su orden de Strahler."""
    global last_watershed_data, root_window, max_strahler_order
    
    if not last_watershed_data["calculated"]:
        messagebox.showinfo("Información", "Primero debe delinear una cuenca para visualizar el orden de Strahler.", parent=root_window)
        return
    
    strahler_order_np = last_watershed_data["strahler_order_data"]
    watershed_mask = last_watershed_data["watershed_mask"]
    rivers_for_strahler_mask = last_watershed_data["rivers_for_strahler_mask"]
    transform = last_watershed_data["transform"]
    dem_display_np = last_watershed_data["dem_data"]
    
    if strahler_order_np is None or not np.any(rivers_for_strahler_mask):
        messagebox.showinfo("Información", "No hay datos de orden de Strahler disponibles o la red de drenaje es muy pequeña.", parent=root_window)
        return
    
    # Crear figura con fondo transparente para mejor visualización
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    # Configurar el título con información sobre el orden máximo
    max_order = int(np.nanmax(strahler_order_np[rivers_for_strahler_mask])) if np.any(strahler_order_np[rivers_for_strahler_mask]) else 0
    if max_order == 0: # Fallback if no valid Strahler order found
        messagebox.showinfo("Información", "No se encontraron órdenes de Strahler válidos para mostrar en el mapa detallado.", parent=root_window)
        plt.close(fig) # Close the empty figure
        return

    fig.suptitle(f"Orden de Strahler de la Red de Drenaje\nOrden máximo: {max_order}", fontsize=16)
    
    # Preparar un DEM simplificado (sombreado de relieve) como fondo
    hillshade = None
    try:
        # Intentar crear un sombreado de relieve si hay datos válidos de DEM
        if dem_display_np is not None:
            # Copia para no modificar el original
            dem_for_hillshade = dem_display_np.copy()
            # Máscara para los valores inválidos
            invalid_mask = np.isnan(dem_for_hillshade) | (dem_for_hillshade == -9999)
            # Rellenar valores inválidos con la media de los válidos para el hillshade
            if np.any(~invalid_mask):
                mean_valid = np.mean(dem_for_hillshade[~invalid_mask])
                dem_for_hillshade[invalid_mask] = mean_valid
                # Calcular hillshade
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)
                hillshade = ls.hillshade(dem_for_hillshade, vert_exag=0.3)
                
                # Obtener extent para el hillshade usando dem_for_hillshade y transform
                hillshade_extent = plotting_extent(dem_for_hillshade, transform)

                # Mostrar hillshade con transparencia
                ax.imshow(hillshade, cmap='gray', alpha=0.3, extent=hillshade_extent)
            else:
                print("No se encontraron valores válidos en el DEM para crear el hillshade.")
    except Exception as e:
        print(f"Error al crear hillshade: {e}")
        # Continuar sin hillshade
    
    # Definir una paleta de colores vibrantes para los diferentes órdenes
    colors = [
        '#1f77b4',  # Azul (orden 1)
        '#ff7f0e',  # Naranja (orden 2)
        '#2ca02c',  # Verde (orden 3)
        '#d62728',  # Rojo (orden 4)
        '#9467bd',  # Morado (orden 5)
        '#8c564b',  # Marrón (orden 6)
        '#e377c2',  # Rosa (orden 7)
        '#7f7f7f',  # Gris (orden 8)
        '#bcbd22',  # Amarillo-verde (orden 9)
        '#17becf'   # Azul claro (orden 10)
    ]
    
    # Extender los colores si tenemos más de 10 órdenes (poco común)
    while max_order > len(colors):
        colors.extend(colors)
    
    # Obtener extent para la visualización (usando el extent del DEM original o el de la cuenca)
    # Se asume que dem_display_np.shape y transform son consistentes con la cuenca
    extent = plotting_extent(dem_display_np, transform)
    
    # Dibujar el contorno de la cuenca
    try:
        shapes_gen = shapes(watershed_mask.astype(rasterio.uint8), mask=watershed_mask, transform=transform)
        all_shapes = [shape(geom) for geom, value in shapes_gen if value == 1]
        
        if all_shapes:
            if len(all_shapes) > 1:
                basin_shape = MultiPolygon(all_shapes)
            else:
                basin_shape = all_shapes[0]
                
            if basin_shape.geom_type == 'Polygon':
                x_poly, y_poly = basin_shape.exterior.xy
                ax.plot(x_poly, y_poly, color='black', linewidth=1.5, solid_capstyle='round', zorder=5)
                for interior in basin_shape.interiors:
                    x_hole, y_hole = interior.xy
                    ax.plot(x_hole, y_hole, color='black', linewidth=1.5, solid_capstyle='round', zorder=5)
            elif basin_shape.geom_type == 'MultiPolygon':
                for poly in basin_shape.geoms:
                    if poly.geom_type == 'Polygon':
                        x_poly, y_poly = poly.exterior.xy
                        ax.plot(x_poly, y_poly, color='black', linewidth=1.5, solid_capstyle='round', zorder=5)
                        for interior in poly.interiors:
                            x_hole, y_hole = interior.xy
                            ax.plot(x_hole, y_hole, color='black', linewidth=1.5, solid_capstyle='round', zorder=5)
    except Exception as e:
        print(f"Error al dibujar contorno de cuenca: {e}")
        # Continuar sin contorno
    
    # Trazar los ríos por orden, sin dilatación para mantenerlos delgados
    legend_handles = []
    
    for order in range(1, max_order + 1):
        # Seleccionar ríos del orden actual
        order_mask = (strahler_order_np == order) & rivers_for_strahler_mask
        
        if np.any(order_mask):
            # Convertir la máscara a coordenadas de píxeles
            river_pixels = np.where(order_mask)
            
            # Convertir a coordenadas geográficas
            river_x, river_y = rasterio.transform.xy(transform, river_pixels[0], river_pixels[1])
            
            # Usamos diferentes grosores según el orden para mejor visibilidad
            # Ajuste de linewidth para que los ríos sean delgados pero distinguibles
            linewidth = 0.5 + (order - 1) * 0.2 # Base 0.5, incrementa 0.2 por orden
            linewidth = min(linewidth, 2.0) # Máximo grosor para evitar que sean demasiado anchos
            
            color = colors[(order-1) % len(colors)]
            
            # Usar scatter para puntos, con tamaño variable para simular grosor de línea
            ax.scatter(river_x, river_y, s=linewidth*10, color=color, alpha=0.9, 
                      edgecolors='none', marker='.', zorder=10+order)
            
            # Añadir a leyenda
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=linewidth*2, 
                                             label=f'Orden {order}'))
            
            # --- MODIFICACIÓN: Añadir una única etiqueta por orden ---
            # Calcular el centroide de los píxeles de este orden
            mean_row = int(np.mean(river_pixels[0]))
            mean_col = int(np.mean(river_pixels[1]))
            
            # Convertir la coordenada del centroide a coordenadas del mapa
            label_x, label_y = rasterio.transform.xy(transform, mean_row, mean_col)
            
            # Añadir el texto con el número de orden en el centroide
            ax.text(label_x, label_y, str(order),
                    fontsize=10, fontweight='bold', ha='center', va='center',
                    color='white',
                    bbox=dict(boxstyle="circle", fc=color, ec="black", lw=0.5, alpha=0.9, pad=0.2),
                    zorder=30) # zorder alto para que esté encima de todo
    
    # Añadir leyenda
    ax.legend(handles=legend_handles, loc='upper right', title="Orden de Strahler", 
              frameon=True, framealpha=0.8)
    
    # Configurar ejes y límites
    ax.set_xlabel("Coordenada X / Longitud")
    ax.set_ylabel("Coordenada Y / Latitud")
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    # MODIFICACIÓN: Rotar etiquetas del eje X para mejor visibilidad
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Añadir una escala de referencia si hay información de CRS
    if last_watershed_data["is_geographic"]:
        pass # Para coordenadas geográficas, no añadimos escala (sería inexacta)
    else:
        try:
            x_range = extent[1] - extent[0]
            scale_length = 10 ** np.floor(np.log10(x_range / 10))
            scale_pos_x = extent[0] + 0.75 * x_range
            scale_pos_y = extent[2] + 0.05 * (extent[3] - extent[2])
            ax.plot([scale_pos_x, scale_pos_x + scale_length], [scale_pos_y, scale_pos_y], 
                    'k-', linewidth=2)
            if scale_length >= 1000:
                ax.text(scale_pos_x + scale_length/2, scale_pos_y + (extent[3]-extent[2])*0.01, 
                        f"{scale_length/1000:.0f} km", ha='center', va='bottom')
            else:
                ax.text(scale_pos_x + scale_length/2, scale_pos_y + (extent[3]-extent[2])*0.01, 
                        f"{scale_length:.0f} m", ha='center', va='bottom')
        except Exception as e:
            print(f"Error al añadir escala: {e}")
    
    # Ajustar el layout y mostrar
    plt.subplots_adjust(bottom=0.15, top=0.9)  # Espacio para el título y las etiquetas rotadas
    plt.tight_layout()
    plt.show()
    
    return

# --- Main Entry Point ---
if __name__ == "__main__":
    root_window = tk.Tk()
    
    create_main_window(
        root=root_window,
        main_callback=main_action,
        load_existing_callback=load_existing_watershed_action,
        manual_coord_callback=manual_coordinate_workflow,
        morph_callback=show_morphometric_indices
        
    )
    
    root_window.mainloop()
