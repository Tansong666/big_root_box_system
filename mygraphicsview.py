import os

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QGuiApplication, QPixmap, QPainter, QImage, QPolygonF, QBrush, QColor, QPen


class myGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.HighQualityAntialiasing |
                            QPainter.SmoothPixmapTransform)
        self.setCacheMode(self.CacheBackground)
        self.setViewportUpdateMode(self.SmartViewportUpdate)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._item = QtWidgets.QGraphicsPixmapItem()  # 放置图像
        self._scene = QtWidgets.QGraphicsScene(self)  # 场景
        self.setScene(self._scene)
        self._scene.addItem(self._item)
        self.pixmap = None
        self._delta = 0.1  # 缩放

        # 画图
        self.temp_polygon = None
        self.drawMode = False
        self.drawing = False
        self.polygon_points = []
        self.point_items = []
        self.line_items = []
        self.movingline = None
        self.polygon_item = None
        self.polygon_items = []
        self.itemAtMouse = None
        self.select_item = None
        self.pen = QPen(QColor("red"), 2)

    def setPixmap(self, pixmap, fitInView=True):
        if isinstance(pixmap, QPixmap):
            self.pixmap = pixmap
        elif isinstance(pixmap, QImage):
            self.pixmap = QPixmap.fromImage(pixmap)
        elif isinstance(pixmap, str) and os.path.isfile(pixmap):
            self.pixmap = QPixmap(pixmap)
        else:
            return
        self._item.setPixmap(self.pixmap)
        self._item.update()
        self.setSceneDims()
        self.update()
        if fitInView:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_a_polygon(self, polygon_points):
        if self.temp_polygon is not None:
            self._scene.removeItem(self.temp_polygon)
        if len(polygon_points) == 4:
            # 将点转换为QPointF
            polygon_points = [QPointF(x, y) for x, y in polygon_points]
            self.temp_polygon = self._scene.addPolygon(QPolygonF(polygon_points), self.pen,
                                                       QBrush(QColor(255, 0, 0, 200)))
        else:
            self.temp_polygon = None

    def clean_items(self):
        if self.temp_polygon is not None:
            self._scene.removeItem(self.temp_polygon)
            self.temp_polygon = None
        if self.movingline is not None:
            self._scene.removeItem(self.movingline)
            self.movingline = None
        if self.polygon_items:
            for item in self.polygon_items:
                self._scene.removeItem(item)
            self.polygon_items = []
        if self.point_items:
            for item in self.point_items:
                self._scene.removeItem(item)
            self.point_items = []
        if self.line_items:
            for item in self.line_items:
                self._scene.removeItem(item)
            self.line_items = []

    def wheelEvent(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale(1 + self._delta, 1 + self._delta)
            else:
                self.scale(1 - self._delta, 1 - self._delta)
        else:
            super().wheelEvent(event)

    def setSceneDims(self):
        if not self.pixmap:
            return
        self.setSceneRect(QRectF(QPointF(0, 0), QPointF(self.pixmap.width(), self.pixmap.height())))

    def fitInView(self, rect: QtCore.QRectF, mode: QtCore.Qt.AspectRatioMode = ...) -> None:
        if not self.pixmap or not self.isVisible():
            return
        if rect is False:
            rect = QRectF(QPointF(0, 0), QPointF(self.pixmap.width(), self.pixmap.height()))
        mode = Qt.KeepAspectRatio
        super().fitInView(rect, mode)

    def setDrawMode(self, mode):
        self.drawMode = mode
        if mode:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        else:
            for item in self.polygon_items:
                self._scene.removeItem(item)
            self.polygon_items = []
            for item in self.point_items:
                self._scene.removeItem(item)
            self.point_items = []
            for item in self.line_items:
                self._scene.removeItem(item)
            self.line_items = []
            self.polygon_points = []
            self.drawing = False
            if self.movingline:
                self._scene.removeItem(self.movingline)
                self.movingline = None
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def get_polygons(self):
        polygons = []
        for item in self.polygon_items:
            polygon = []
            for point in list(item.polygon()):
                point = self._item.mapFromItem(item, point)
                x, y = int(point.x()), int(point.y())
                polygon.append([x, y])
            polygons.append(polygon)
            self._scene.removeItem(item)
        self.polygon_items = []
        return polygons

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_R:
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        elif event.key() == Qt.Key_Delete:
            if self.drawing and self.drawMode:
                if len(self.polygon_points) == 1:
                    self.drawing = False
                    self._scene.removeItem(self.movingline)
                    self.movingline = None
                    self._scene.removeItem(self.point_items[-1])
                    self.point_items.pop()
                    self.polygon_points.pop()
                else:
                    mouse_pos = self.movingline.line().p2()
                    self._scene.removeItem(self.movingline)
                    self.polygon_points.pop()
                    self._scene.removeItem(self.point_items.pop())
                    self._scene.removeItem(self.line_items.pop())
                    self.movingline = self._scene.addLine(self.polygon_points[-1].x(), self.polygon_points[-1].y(),
                                                          mouse_pos.x(), mouse_pos.y(), self.pen)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if not self.drawMode:
            return
        point = self.mapToScene(event.pos())
        x, y = int(point.x()), int(point.y())
        point = QPointF(x, y)
        if event.button() == Qt.LeftButton:
            if self.select_item:
                self.select_item = None
                return
            if not self.drawing:
                self.polygon_points.append(point)
                self.first_point = point
                ellipse = self._scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, self.pen)
                self.point_items.append(ellipse)
                self.movingline = self._scene.addLine(point.x(), point.y(), point.x(), point.y(), self.pen)
                self.drawing = True
            else:
                dx = abs(point.x() - self.first_point.x())
                dy = abs(point.y() - self.first_point.y())
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if distance > 3:
                    last_point = self.polygon_points[-1]
                    self.polygon_points.append(point)
                    line = self._scene.addLine(last_point.x(), last_point.y(), point.x(), point.y(), self.pen)
                    self.line_items.append(line)
                    ellipse = self._scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, self.pen)
                    self.point_items.append(ellipse)
                    self.movingline.setLine(point.x(), point.y(), point.x(), point.y())
                else:
                    self.drawing = False
                    self._scene.removeItem(self.movingline)
                    self.movingline = None
                    polygon_item = self._scene.addPolygon(QPolygonF(self.polygon_points), self.pen,
                                                          QBrush(QColor(255, 0, 0, 200)))
                    self.polygon_items.append(polygon_item)
                    self.polygon_points = []
                    for item in self.point_items:
                        self._scene.removeItem(item)
                    self.point_items = []
                    for item in self.line_items:
                        self._scene.removeItem(item)
                    self.line_items = []
        if event.button() == Qt.RightButton:
            if self.drawing:
                if len(self.polygon_points) == 1:
                    self.drawing = False
                    self._scene.removeItem(self.movingline)
                    self.movingline = None
                    self._scene.removeItem(self.point_items[-1])
                    self.point_items.pop()
                    self.polygon_points.pop()
                else:
                    mouse_pos = self.movingline.line().p2()
                    self._scene.removeItem(self.movingline)
                    self.polygon_points.pop()
                    self._scene.removeItem(self.point_items.pop())
                    self._scene.removeItem(self.line_items.pop())
                    self.movingline = self._scene.addLine(self.polygon_points[-1].x(), self.polygon_points[-1].y(),
                                                          mouse_pos.x(), mouse_pos.y(), self.pen)
            else:
                if self.itemAtMouse:
                    self._scene.removeItem(self.itemAtMouse)
                    self.polygon_items.remove(self.itemAtMouse)
                    self.itemAtMouse = None

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        point = self.mapToScene(event.pos())
        x, y = int(point.x()), int(point.y())
        point = QPointF(x, y)
        if event.button() == Qt.LeftButton:
            if not self.drawing and self.itemAtMouse:
                self.select_item = self.itemAtMouse
                self.old_pos = point

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        point = self.mapToScene(event.pos())
        x, y = int(point.x()), int(point.y())
        point = QPointF(x, y)
        if self.drawing:
            self.movingline.setLine(self.polygon_points[-1].x(), self.polygon_points[-1].y(), point.x(), point.y())
        else:
            itemAtMouse = self._scene.itemAt(point, self.transform())
            if itemAtMouse and itemAtMouse != self.itemAtMouse:
                if self.itemAtMouse:
                    self.itemAtMouse.setPen(self.pen)
                if itemAtMouse in self.polygon_items:
                    itemAtMouse.setPen(QPen(QColor(0, 255, 0, 125), 4))
                else:
                    itemAtMouse = None
                self.itemAtMouse = itemAtMouse
            if self.select_item:
                d = point - self.old_pos
                dx, dy = d.x(), d.y()
                x, y, w, h = self._item.mapRectFromItem(self.select_item, self.select_item.boundingRect()).getRect()
                if x + dx < 0 or x + dx + w > self._item.boundingRect().width():
                    d.setX(0)
                if y + dy < 0 or y + dy + h > self._item.boundingRect().height():
                    d.setY(0)
                self.select_item.setPos(self.select_item.pos() + d)
                self.old_pos = point
