"""主窗口"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QSplitter, QStatusBar, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
import traceback
import numpy as np
import cv2

from .control_panel import ControlPanel
from .image_view import ImageView
from .result_panel import ResultPanel
from ..model import ModelInference
from ..pipeline import KMPipeline
from ..io import load_image, save_image, export_all_curves, setup_logger
from ..utils import draw_curves_on_image, create_mask_visualization


class ProcessThread(QThread):
    """处理线程 - Color-first默认"""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, image_path, checkpoint_path, x_max, output_dir, roi_mode, manual_roi=None):
        super().__init__()
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.x_max = x_max
        self.output_dir = output_dir
        self.roi_mode = roi_mode
        self.manual_roi = manual_roi

    def run(self):
        try:
            from ..pipeline.color_extract import extract_colored_curves
            from ..utils import draw_curves_on_image

            # 加载图像
            self.progress.emit("加载图像...")
            image = load_image(self.image_path)
            h, w = image.shape[:2]

            # 确定ROI
            if self.roi_mode == 'manual' and self.manual_roi is not None:
                roi = self.manual_roi
            elif self.roi_mode == 'auto':
                self.progress.emit("自动检测ROI...")
                from ..pipeline.preprocess import auto_detect_roi
                roi = auto_detect_roi(image)
            else:  # full
                roi = (0, 0, w, h)

            self.progress.emit(f"ROI: {roi}")

            # Color-first提取
            self.progress.emit("提取彩色曲线...")
            result = extract_colored_curves(image, roi=roi, n_colors=5)

            # 保存结果
            self.progress.emit("保存结果...")
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 保存ROI裁剪
            save_image(result['roi_image'], str(output_path / "roi_crop.png"))

            # 保存color masks
            for i, mask in enumerate(result['color_masks']):
                save_image(mask, str(output_path / "mask_color_{}.png".format(i+1)))

            # 保存separated masks
            for i, mask in enumerate(result['separated_masks']):
                save_image(mask, str(output_path / "mask_component_{}.png".format(i+1)))

            # 保存ROI局部结果
            if len(result['pixel_paths_roi']) > 0:
                result_roi = draw_curves_on_image(result['roi_image'], result['pixel_paths_roi'])
                save_image(result_roi, str(output_path / "result_roi.png"))

            # 保存全图结果
            if len(result['pixel_paths_global']) > 0:
                result_global = draw_curves_on_image(result['original_image'], result['pixel_paths_global'])
                save_image(result_global, str(output_path / "result_global.png"))

                # 保存像素坐标CSV
                for i, path in enumerate(result['pixel_paths_global']):
                    csv_path = output_path / "curve_pixels_{}.csv".format(i+1)
                    np.savetxt(csv_path, path, delimiter=',', header='x,y', comments='', fmt='%d')

            # 保存debug overlay
            if len(result['pixel_paths_roi']) > 0:
                debug_overlay = result['roi_image'].copy()
                colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
                for i, path in enumerate(result['pixel_paths_roi']):
                    color = colors[i % len(colors)]
                    for j in range(len(path)-1):
                        cv2.line(debug_overlay, tuple(path[j]), tuple(path[j+1]), color, 2)
                save_image(debug_overlay, str(output_path / "debug_overlay.png"))

            # 保存处理信息
            info_lines = [
                f"处理时间: {np.datetime64('now')}",
                f"ROI: {result['roi']}",
                f"颜色数: {result['stats']['n_colors']}",
                f"连通域数: {result['stats']['n_components']}",
                f"最终曲线数: {result['num_curves']}",
                f"前景像素: {result['stats'].get('foreground_pixels', 'N/A')}"
            ]
            with open(output_path / "process.log", 'w', encoding='utf-8') as f:
                f.write('\n'.join(info_lines))

            self.progress.emit("完成!")
            self.finished.emit(result)

        except Exception as e:
            print(f"\n[错误] {str(e)}")
            print(traceback.format_exc())
            self.error.emit(f"处理失败: {str(e)}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KM生存曲线提取工具")
        self.resize(1400, 900)

        self.current_image = None
        self.current_result = None
        self.process_thread = None
        self.manual_roi = None  # 手动ROI

        self.setup_ui()
        self.setup_logger()

    def setup_ui(self):
        """设置UI"""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        # 左侧控制面板
        self.control_panel = ControlPanel()
        self.control_panel.run_clicked.connect(self.on_run)
        self.control_panel.export_clicked.connect(self.on_export)
        self.control_panel.image_selected.connect(self.on_image_selected)
        self.control_panel.roi_mode_changed.connect(self.on_roi_mode_changed)

        # 中间图像显示
        self.image_view = ImageView()
        self.image_view.roi_selected.connect(self.on_roi_selected)

        # 右侧结果面板
        self.result_panel = ResultPanel()

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.image_view)
        splitter.addWidget(self.result_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        layout.addWidget(splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def setup_logger(self):
        """设置日志"""
        self.logger = setup_logger("KM_App")

    def on_roi_mode_changed(self, mode: str):
        """ROI模式变化"""
        self.image_view.set_roi_mode(mode)
        self.manual_roi = None
        self.status_bar.showMessage(f"ROI模式: {mode}")

    def on_roi_selected(self, roi: tuple):
        """ROI框选完成"""
        self.manual_roi = roi
        self.status_bar.showMessage(f"ROI已选择: {roi}")
        self.result_panel.add_log(f"手动ROI: {roi}")

    def on_run(self):
        """运行处理"""
        # 获取参数
        image_path = self.control_panel.get_image_path()
        checkpoint_path = self.control_panel.get_checkpoint_path()
        x_max = self.control_panel.get_x_max()
        output_dir = self.control_panel.get_output_dir()
        roi_mode = self.control_panel.get_roi_mode()

        if not image_path or not Path(image_path).exists():
            QMessageBox.warning(self, "错误", "请选择有效的图像文件")
            return

        # 模型文件不再强制要求（color-first不需要）
        # if not checkpoint_path or not Path(checkpoint_path).exists():
        #     QMessageBox.warning(self, "错误", "请选择有效的模型文件")
        #     return

        # 检查手动ROI
        if roi_mode == 'manual' and self.manual_roi is None:
            QMessageBox.warning(self, "提示", "请先在图像上框选ROI区域")
            return

        # 加载图像显示
        self.current_image = load_image(image_path)
        self.image_view.set_image(self.current_image)

        # 启动处理线程
        self.process_thread = ProcessThread(
            image_path, checkpoint_path, x_max, output_dir,
            roi_mode, self.manual_roi
        )
        self.process_thread.progress.connect(self.on_progress)
        self.process_thread.finished.connect(self.on_finished)
        self.process_thread.error.connect(self.on_error)
        self.process_thread.start()

        self.control_panel.set_enabled(False)

    def on_progress(self, message):
        """进度更新"""
        self.status_bar.showMessage(message)
        self.result_panel.add_log(message)

    def on_finished(self, result):
        """处理完成"""
        self.current_result = result
        self.control_panel.set_enabled(True)
        self.status_bar.showMessage(f"完成! 提取到 {result['num_curves']} 条曲线")

        # 显示全图结果（优先）
        if 'pixel_paths_global' in result and len(result['pixel_paths_global']) > 0:
            from ..utils import draw_curves_on_image
            result_image = draw_curves_on_image(result['original_image'], result['pixel_paths_global'])
            self.image_view.set_image(result_image)
        elif 'pixel_paths_roi' in result and len(result['pixel_paths_roi']) > 0:
            # 回退到ROI结果
            from ..utils import draw_curves_on_image
            result_image = draw_curves_on_image(result['roi_image'], result['pixel_paths_roi'])
            self.image_view.set_image(result_image)

        # 更新结果面板
        self.result_panel.set_result(result)

        QMessageBox.information(self, "成功", f"处理完成!\n提取到 {result['num_curves']} 条曲线")

    def on_error(self, error_msg):
        """处理错误"""
        self.control_panel.set_enabled(True)
        self.status_bar.showMessage("处理失败")
        self.result_panel.add_log(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)

    def on_export(self):
        """导出结果"""
        if self.current_result is None:
            QMessageBox.warning(self, "提示", "没有可导出的结果")
            return

        output_dir = self.control_panel.get_output_dir()
        QMessageBox.information(self, "成功", f"结果已保存到:\n{output_dir}")

    def on_image_selected(self, image_path: str):
        """图像选择后预览"""
        try:
            from ..io import load_image
            image = load_image(image_path)
            self.image_view.set_image(image)
            self.status_bar.showMessage(f"已加载图像: {Path(image_path).name}")
        except Exception as e:
            self.status_bar.showMessage(f"加载图像失败: {e}")
            QMessageBox.warning(self, "错误", f"无法加载图像:\n{e}")
