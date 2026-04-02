"""结果导出"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


def export_curve_to_csv(chart_coords: np.ndarray, output_path: str, curve_name: str = "curve"):
    """导出单条曲线到CSV"""
    if len(chart_coords) == 0:
        return

    df = pd.DataFrame(chart_coords, columns=['Time', 'Survival_Rate'])
    df['Curve'] = curve_name

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format='%.4f')


def export_all_curves(chart_coords_list: List[np.ndarray], output_dir: str, base_name: str = "curve"):
    """导出所有曲线"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, coords in enumerate(chart_coords_list):
        if len(coords) > 0:
            curve_name = f"{base_name}_{i+1}"
            csv_path = output_path / f"{curve_name}.csv"
            export_curve_to_csv(coords, str(csv_path), curve_name)

    # 合并导出
    if chart_coords_list:
        all_data = []
        for i, coords in enumerate(chart_coords_list):
            if len(coords) > 0:
                df = pd.DataFrame(coords, columns=['Time', 'Survival_Rate'])
                df['Curve'] = f"curve_{i+1}"
                all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined_path = output_path / "all_curves.csv"
            combined.to_csv(combined_path, index=False, float_format='%.4f')
