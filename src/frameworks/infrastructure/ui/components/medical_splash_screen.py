import logging
from typing import Optional, Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QSplashScreen, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QPixmap, QFont, QPainter, QColor, QBrush, QPen
from PyQt6.QtSvg import QSvgRenderer

logger = logging.getLogger(__name__)


class MedicalSplashScreen(QSplashScreen):
    initialization_completed = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self,
                 app_name: str = "Medical Imaging Workstation",
                 app_version: str = "2.0.0",
                 logo_path: Optional[str] = None,
                 min_display_time: int = 3000):  
       
        pixmap = self._create_medical_splash_pixmap(app_name, app_version, logo_path)
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.SplashScreen)

        self._app_name = app_name
        self._app_version = app_version
        self._min_display_time = min_display_time

        self._initialization_steps = []
        self._current_step = 0
        self._progress_value = 0
        self._status_message = "Iniciando aplicación médica..."

        self._min_time_timer = QTimer()
        self._min_time_timer.setSingleShot(True)
        self._min_time_timer.timeout.connect(self._on_min_time_reached)

        self._min_time_reached = False
        self._initialization_completed = False

        self._setup_medical_fonts()

        logger.info(f"Medical splash screen creado: {app_name} v{app_version}")

    def _create_medical_splash_pixmap(self,
                                    app_name: str,
                                    app_version: str,
                                    logo_path: Optional[str] = None) -> QPixmap:
    
        width, height = 600, 400  
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(16, 20, 24)) 

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._paint_minimal_background(painter, width, height)

        if logo_path and Path(logo_path).exists():
            self._paint_medical_logo_centered(painter, logo_path, width, height)
        else:
            self._paint_minimal_title(painter, app_name, width, height)

        self._paint_progress_area(painter, width, height)

        painter.end()
        return pixmap

    def _paint_minimal_background(self, painter: QPainter, width: int, height: int) -> None:
        pen = QPen(QColor(60, 60, 60), 1) 
        painter.setPen(pen)
        painter.drawRect(1, 1, width-2, height-2)

    def _paint_medical_logo_centered(self, painter: QPainter, logo_path: str, width: int, height: int) -> None:
        try:
            margin = 15  
            progress_space = 60  

            logo_width = width - (2 * margin)
            logo_height = height - progress_space - (2 * margin)

            x = margin
            y = margin

            if logo_path.endswith('.svg'):
                svg_renderer = QSvgRenderer(logo_path)
                if svg_renderer.isValid():
                    svg_renderer.render(painter, x, y, logo_width, logo_height)
                else:
                    logger.warning(f"SVG inválido: {logo_path}")
            else:
                logo_pixmap = QPixmap(logo_path)
                if not logo_pixmap.isNull():
                    scaled_logo = logo_pixmap.scaled(
                        logo_width, logo_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    logo_x = x + (logo_width - scaled_logo.width()) // 2
                    logo_y = y + (logo_height - scaled_logo.height()) // 2
                    painter.drawPixmap(logo_x, logo_y, scaled_logo)
                else:
                    logger.warning(f"Imagin inválida: {logo_path}")
        except Exception as e:
            logger.warning(f"No se pudo cargar logo médico: {e}")

    def _paint_minimal_title(self, painter: QPainter, app_name: str, width: int, height: int) -> None:
        font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        simple_title = "Deep Prostate"
        title_rect = painter.fontMetrics().boundingRect(simple_title)
        x = (width - title_rect.width()) // 2
        y = height // 2
        painter.drawText(x, y, simple_title)

    def _paint_progress_area(self, painter: QPainter, width: int, height: int) -> None:
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(QColor(189, 195, 199))
        status_text = "Preparando aplicación médica..."
        status_rect = painter.fontMetrics().boundingRect(status_text)
        x = (width - status_rect.width()) // 2
        y = height - 45
        painter.drawText(x, y, status_text)

    def _setup_medical_fonts(self) -> None:
        self._title_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        self._status_font = QFont("Segoe UI", 9)
        self._progress_font = QFont("Segoe UI", 8)

    def set_initialization_steps(self, steps: list[str]) -> None:
        self._initialization_steps = steps
        self._current_step = 0
        self._progress_value = 0
        logger.info(f"Pasos of inicialización configurados: {len(steps)} pasos")

    def show_and_start_timer(self) -> None:
        self.show()
        self._min_time_timer.start(self._min_display_time)
        logger.info(f"Splash screen mostrado by mínimo {self._min_display_time}ms")

    def update_progress(self, step_name: str, progress: int = None) -> None:
        self._status_message = step_name

        if progress is not None:
            self._progress_value = progress
        else:
            if self._initialization_steps:
                self._progress_value = int((self._current_step / len(self._initialization_steps)) * 100)
                self._current_step += 1

        self._update_splash_display()
        logger.debug(f"Progreso actualizado: {step_name} ({self._progress_value}%)")

    def _update_splash_display(self) -> None:
        message = f"{self._status_message}\n"
        if self._initialization_steps and self._current_step <= len(self._initialization_steps):
            message += f"Paso {self._current_step}/{len(self._initialization_steps)}"

        self.showMessage(
            message,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor(189, 195, 199)
        )

        self.repaint()
        if hasattr(self, '_qt_app'):
            self._qt_app.processEvents()

    def set_qt_application(self, qt_app) -> None:
        self._qt_app = qt_app

    def mark_initialization_completed(self) -> None:
        self._initialization_completed = True
        self.update_progress("Inicialización completada", 100)
        logger.info("Inicialización marcada como completada")
        self._check_ready_to_close()

    def _on_min_time_reached(self) -> None:
        self._min_time_reached = True
        logger.info("Tiempo mínimo of splash screen alcanzado")
        self._check_ready_to_close()

    def _check_ready_to_close(self) -> None:
        if self._min_time_reached and self._initialization_completed:
            logger.info("Splash screen listo for cerrar")
            self.initialization_completed.emit()

            # Timer for cierre suave
            QTimer.singleShot(800, self._close_splash) 

    def _close_splash(self) -> None:
        self.close_requested.emit()
        self.close()
        logger.info("Splash screen cerrado")

    def mousePressEvent(self, event) -> None:
        """Permite cerrar el splash haciendo clic (for desarrollo)."""
        if event.button() == Qt.MouseButton.LeftButton:
            logger.debug("Cierre manual of the splash screen solicitado")
            self._close_splash()
        super().mousePressEvent(event)


class SplashScreenManager(QObject):
    splash_closed = pyqtSignal()

    def __init__(self,
                 app_name: str = "Deep Prostate",
                 app_version: str = "2.0.0",
                 logo_path: Optional[str] = None):
        super().__init__()

        self._splash = MedicalSplashScreen(app_name, app_version, logo_path)
        self._splash.initialization_completed.connect(self._on_splash_ready)
        self._splash.close_requested.connect(self.splash_closed.emit)

        self._default_steps = [
            "Setting up medical environment...",
            "Validating medical prerequisites...",
            "Initializing medical services...",
            "Creando componentes of UI...",
            "Integrating complete system...",
            "Preparing medical workstation..."
        ]

        self._splash.set_initialization_steps(self._default_steps)

    def show_splash(self, qt_app) -> None:
        self._splash.set_qt_application(qt_app)
        self._splash.show_and_start_timer()

    def update_progress(self, step_name: str, progress: int = None) -> None:
        self._splash.update_progress(step_name, progress)

    def complete_initialization(self) -> None:
        self._splash.mark_initialization_completed()

    def _on_splash_ready(self) -> None:
        logger.info("Splash screen manager: Inicialización completada")

    def set_custom_steps(self, steps: list[str]) -> None:
        self._splash.set_initialization_steps(steps)

    def get_splash_widget(self) -> MedicalSplashScreen:
        return self._splash