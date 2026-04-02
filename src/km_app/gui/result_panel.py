"""结果面板"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QLabel,
                               QGroupBox, QPushButton, QComboBox)
from PySide6.QtCore import Qt


class ResultPanel(QWidget):
    """结果面板"""

    def __init__(self):
        super().__init__()
        self.current_result = None
        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 结果信息
        info_group = QGroupBox("提取结果")
        info_layout = QVBoxLayout(info_group)

        self.info_label = QLabel("等待处理...")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        layout.addWidget(info_group)

        # 日志
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)

        btn_clear_log = QPushButton("清除日志")
        btn_clear_log.clicked.connect(self.log_text.clear)
        log_layout.addWidget(btn_clear_log)

        layout.addWidget(log_group)

        layout.addStretch()

    def set_result(self, result: dict):
        """设置结果"""
        self.current_result = result

        num_curves = result.get('num_curves', 0)
        info_text = f"提取到 {num_curves} 条曲线\n"
        info_text += f"ROI: {result.get('roi', 'N/A')}\n"

        self.info_label.setText(info_text)
        self.add_log(f"成功提取 {num_curves} 条曲线")

    def add_log(self, message: str):
        """添加日志"""
        self.log_text.append(message)
