"""控制面板"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                               QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox,
                               QComboBox, QRadioButton, QButtonGroup)
from PySide6.QtCore import Signal
from pathlib import Path
from ..config import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_X_MAX


class ControlPanel(QWidget):
    """控制面板"""
    run_clicked = Signal()
    export_clicked = Signal()
    image_selected = Signal(str)
    roi_mode_changed = Signal(str)  # 新增：ROI模式变化信号

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 图像选择
        img_group = QGroupBox("图像文件")
        img_layout = QVBoxLayout(img_group)

        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("选择KM曲线图像...")
        img_layout.addWidget(self.image_path_edit)

        btn_browse_image = QPushButton("浏览...")
        btn_browse_image.clicked.connect(self.browse_image)
        img_layout.addWidget(btn_browse_image)

        layout.addWidget(img_group)

        # ROI模式选择
        roi_group = QGroupBox("ROI模式")
        roi_layout = QVBoxLayout(roi_group)

        self.roi_button_group = QButtonGroup(self)

        self.roi_auto_radio = QRadioButton("自动检测")
        self.roi_auto_radio.setChecked(True)
        self.roi_button_group.addButton(self.roi_auto_radio, 0)
        roi_layout.addWidget(self.roi_auto_radio)

        self.roi_manual_radio = QRadioButton("手动框选")
        self.roi_button_group.addButton(self.roi_manual_radio, 1)
        roi_layout.addWidget(self.roi_manual_radio)

        self.roi_full_radio = QRadioButton("全图（不推荐）")
        self.roi_button_group.addButton(self.roi_full_radio, 2)
        roi_layout.addWidget(self.roi_full_radio)

        # 连接信号
        self.roi_button_group.buttonClicked.connect(self.on_roi_mode_changed)

        layout.addWidget(roi_group)

        # 模型选择
        model_group = QGroupBox("模型文件")
        model_layout = QVBoxLayout(model_group)

        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setText(str(DEFAULT_MODEL_PATH))
        self.checkpoint_path_edit.setPlaceholderText("选择模型权重文件...")
        model_layout.addWidget(self.checkpoint_path_edit)

        btn_browse_model = QPushButton("浏览...")
        btn_browse_model.clicked.connect(self.browse_checkpoint)
        model_layout.addWidget(btn_browse_model)

        layout.addWidget(model_group)

        # 参数设置
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)

        param_layout.addWidget(QLabel("X轴最大时间:"))
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(1.0, 1000.0)
        self.x_max_spin.setValue(DEFAULT_X_MAX)
        self.x_max_spin.setSuffix(" 月")
        param_layout.addWidget(self.x_max_spin)

        layout.addWidget(param_group)

        # 输出目录
        output_group = QGroupBox("输出目录")
        output_layout = QVBoxLayout(output_group)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(DEFAULT_OUTPUT_DIR))
        output_layout.addWidget(self.output_dir_edit)

        btn_browse_output = QPushButton("浏览...")
        btn_browse_output.clicked.connect(self.browse_output)
        output_layout.addWidget(btn_browse_output)

        layout.addWidget(output_group)

        # 运行按钮
        self.btn_run = QPushButton("运行提取")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        self.btn_run.clicked.connect(self.run_clicked.emit)
        layout.addWidget(self.btn_run)

        # 导出按钮
        self.btn_export = QPushButton("导出结果")
        self.btn_export.clicked.connect(self.export_clicked.emit)
        layout.addWidget(self.btn_export)

        layout.addStretch()

    def on_roi_mode_changed(self):
        """ROI模式变化"""
        mode = self.get_roi_mode()
        self.roi_mode_changed.emit(mode)

    def browse_image(self):
        """浏览图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        if file_path:
            self.image_path_edit.setText(file_path)
            self.image_selected.emit(file_path)

    def browse_checkpoint(self):
        """浏览模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "",
            "模型文件 (*.pth *.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.checkpoint_path_edit.setText(file_path)

    def browse_output(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_image_path(self) -> str:
        return self.image_path_edit.text()

    def get_checkpoint_path(self) -> str:
        return self.checkpoint_path_edit.text()

    def get_x_max(self) -> float:
        return self.x_max_spin.value()

    def get_output_dir(self) -> str:
        return self.output_dir_edit.text()

    def get_roi_mode(self) -> str:
        """获取ROI模式"""
        if self.roi_auto_radio.isChecked():
            return 'auto'
        elif self.roi_manual_radio.isChecked():
            return 'manual'
        else:
            return 'full'

    def set_enabled(self, enabled: bool):
        """设置控件启用状态"""
        self.btn_run.setEnabled(enabled)
        self.image_path_edit.setEnabled(enabled)
        self.checkpoint_path_edit.setEnabled(enabled)
        self.x_max_spin.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled)
