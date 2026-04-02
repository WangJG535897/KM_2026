"""图像显示"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
import numpy as np
import cv2


class ImageView(QWidget):
    """图像显示组件 - 支持ROI框选"""
    roi_selected = Signal(tuple)  # 发射ROI坐标 (x1, y1, x2, y2)

    def __init__(self):
        super().__init__()
        self.current_image = None
        self.original_image = None
        self.roi_mode = 'auto'
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_dragging = False  # 拖拽状态
        self.scale_factor = 1.0
        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background-color: #2b2b2b; }")

        # 图像标签
        self.image_label = InteractiveLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; color: #888; font-size: 14px; }")
        self.image_label.setMinimumSize(400, 300)

        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)

    def set_roi_mode(self, mode: str):
        """设置ROI模式"""
        self.roi_mode = mode
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_dragging = False
        if self.original_image is not None:
            self.update_display()

    def set_image(self, image: np.ndarray):
        """设置显示图像"""
        if image is None:
            self.image_label.clear()
            self.image_label.setText("未加载图像")
            return

        self.original_image = image.copy()
        self.current_image = image.copy()
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_dragging = False
        self.update_display()

    def update_display(self):
        """更新显示"""
        if self.original_image is None:
            return

        display_image = self.current_image.copy()

        # 绘制ROI框
        if self.roi_rect is not None:
            x1, y1, x2, y2 = self.roi_rect
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, "ROI", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 转换为RGB
        if len(display_image.shape) == 2:
            image_rgb = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

        h, w, c = image_rgb.shape

        # 转换为QImage
        bytes_per_line = c * w
        q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 转换为QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # 缩放以适应窗口大小
        label_size = self.image_label.size()
        if label_size.width() > 100 and label_size.height() > 100:
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.scale_factor = scaled_pixmap.width() / w
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setPixmap(pixmap)
            self.scale_factor = 1.0

    def get_roi(self):
        """获取当前ROI"""
        return self.roi_rect

    def handle_mouse_press(self, event: QMouseEvent):
        """处理鼠标按下（由InteractiveLabel调用）"""
        # 只响应左键
        if event.button() != Qt.LeftButton:
            return

        if self.roi_mode != 'manual' or self.original_image is None:
            return

        # 转换到原图坐标
        orig_x, orig_y = self._screen_to_image_coords(event.pos())
        if orig_x is not None:
            # 开始新的拖拽
            self.is_dragging = True
            self.roi_start = (orig_x, orig_y)
            self.roi_end = None
            self.roi_rect = None

    def handle_mouse_move(self, event: QMouseEvent):
        """处理鼠标移动（由InteractiveLabel调用）"""
        # 只有在拖拽状态才响应
        if not self.is_dragging or self.roi_start is None:
            return

        # 转换到原图坐标
        orig_x, orig_y = self._screen_to_image_coords(event.pos())
        if orig_x is not None:
            self.roi_end = (orig_x, orig_y)

            # 更新ROI矩形
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            self.roi_rect = (x1, y1, x2, y2)

            self.update_display()

    def handle_mouse_release(self, event: QMouseEvent):
        """处理鼠标释放（由InteractiveLabel调用）"""
        # 只响应左键
        if event.button() != Qt.LeftButton:
            return

        if not self.is_dragging or self.roi_start is None or self.roi_end is None:
            # 结束拖拽状态
            self.is_dragging = False
            return

        # 检查ROI大小
        x1, y1, x2, y2 = self.roi_rect
        width = x2 - x1
        height = y2 - y1

        if width < 50 or height < 50:
            # ROI太小，取消
            self.roi_start = None
            self.roi_end = None
            self.roi_rect = None
            self.is_dragging = False
            self.update_display()
            return

        # 结束拖拽，固定ROI
        self.is_dragging = False

        # 发射ROI信号
        self.roi_selected.emit(self.roi_rect)

    def _screen_to_image_coords(self, pos: QPoint):
        """屏幕坐标转原图坐标"""
        if not self.image_label.pixmap():
            return None, None

        pixmap = self.image_label.pixmap()
        label_rect = self.image_label.rect()

        # 计算pixmap在label中的位置（居中）
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        x_offset = (label_rect.width() - pixmap_width) // 2
        y_offset = (label_rect.height() - pixmap_height) // 2

        # 检查点是否在pixmap内
        if (pos.x() < x_offset or pos.x() >= x_offset + pixmap_width or
            pos.y() < y_offset or pos.y() >= y_offset + pixmap_height):
            return None, None

        # 转换到pixmap坐标
        pixmap_x = pos.x() - x_offset
        pixmap_y = pos.y() - y_offset

        # 转换到原图坐标
        h, w = self.original_image.shape[:2]
        orig_x = int(pixmap_x / self.scale_factor)
        orig_y = int(pixmap_y / self.scale_factor)

        # 边界检查
        orig_x = max(0, min(orig_x, w - 1))
        orig_y = max(0, min(orig_y, h - 1))

        return orig_x, orig_y


class InteractiveLabel(QLabel):
    """可交互的标签 - 转发鼠标事件"""
    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下"""
        self.parent_view.handle_mouse_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动"""
        self.parent_view.handle_mouse_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放"""
        self.parent_view.handle_mouse_release(event)
        super().mouseReleaseEvent(event)

