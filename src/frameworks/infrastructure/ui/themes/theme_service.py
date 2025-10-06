
import logging
from typing import Dict, Any, Optional
from enum import Enum

from .medical_theme import MedicalColorPalette, MedicalThemeManager


class ComponentType(Enum):
    BUTTON = "button"
    DELETE_BUTTON = "delete_button"
    ACTION_BUTTON = "action_button"
    TOGGLE_BUTTON = "toggle_button"
    ICON_BUTTON = "icon_button"
    MENU_BUTTON = "menu_button"
    PANEL_BUTTON = "panel_button"
    SIDEBAR = "sidebar"
    PANEL = "panel"
    GROUP_BOX = "group_box"
    TAB_WIDGET = "tab_widget"
    PROGRESS_BAR = "progress_bar"
    SLIDER = "slider"
    INPUT_FIELD = "input_field"
    LIST_WIDGET = "list_widget"
    TABLE_WIDGET = "table_widget"
    TOOLBAR = "toolbar"
    STATUS_BAR = "status_bar"
    MEDICAL_IMAGE_CANVAS = "medical_image_canvas"
    DIALOG = "dialog"
    SCROLL_AREA = "scroll_area"
    TREE_VIEW = "tree_view"


class StyleVariant(Enum):
    DEFAULT = "default"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
    MEDICAL = "medical"
    COMPACT = "compact"
    LARGE = "large"


class ThemeService:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._style_cache: Dict[str, str] = {}
        self._base_theme = MedicalThemeManager.get_radiology_dark_theme()
        self._component_styles = self._initialize_component_styles()        
        self._logger.info("ThemeService initialized with medical theme")
    
    def get_application_theme(self) -> str:
        return self._base_theme
    
    def get_component_style(
        self, 
        component_type: ComponentType, 
        variant: StyleVariant = StyleVariant.DEFAULT,
        custom_properties: Optional[Dict[str, str]] = None
    ) -> str:
        cache_key = f"{component_type.value}_{variant.value}"
        
        if cache_key in self._style_cache:
            style = self._style_cache[cache_key]
        else:
            style = self._build_component_style(component_type, variant)
            self._style_cache[cache_key] = style
        
        if custom_properties:
            style = self._apply_custom_properties(style, custom_properties)
        
        return style
    
    def get_medical_button_style(
        self,
        button_type: str = "default",
        size: str = "normal"
    ) -> str:
        base_style = """

        QPushButton {{
            background-color: {bg_color};
            color: {text_color};
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: {padding};
            font-weight: 500;
            min-height: {min_height};
            font-size: {font_size};
        }}

        QPushButton:hover {{
            background-color: {hover_bg} !important;
            border-color: {hover_border} !important;
            color: white !important;
        }}

        QPushButton:pressed {{
            background-color: {pressed_bg};
            border-color: {pressed_border};
        }}

        QPushButton:checked {{
            background-color: {checked_bg};
            border-color: {checked_border};
            color: {checked_text};
        }}

        QPushButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_text};
            border-color: {disabled_border};
        }}
        """
        size_configs = {
            "compact": {"padding": "4px 8px", "min_height": "16px", "font_size": "11px"},
            "normal": {"padding": "6px 12px", "min_height": "20px", "font_size": "12px"},
            "large": {"padding": "8px 16px", "min_height": "24px", "font_size": "13px"}
        }
        
        type_configs = {
            "default": {
                "bg_color": MedicalColorPalette.BACKGROUND_LIGHT,
                "text_color": MedicalColorPalette.TEXT_PRIMARY,
                "border_color": MedicalColorPalette.BORDER_NORMAL,
                "hover_bg": MedicalColorPalette.ACCENT_BLUE,
                "hover_border": MedicalColorPalette.BORDER_ACTIVE,
                "pressed_bg": MedicalColorPalette.ACCENT_BLUE,
                "pressed_border": MedicalColorPalette.BORDER_ACTIVE,
                "checked_bg": MedicalColorPalette.ACCENT_ORANGE,
                "checked_border": MedicalColorPalette.ACCENT_ORANGE,
                "checked_text": "white",
                "disabled_bg": MedicalColorPalette.BACKGROUND_MEDIUM,
                "disabled_text": MedicalColorPalette.TEXT_DISABLED,
                "disabled_border": MedicalColorPalette.BORDER_SUBTLE
            },
            "primary": {
                "bg_color": MedicalColorPalette.ACCENT_BLUE,
                "text_color": "white",
                "border_color": MedicalColorPalette.ACCENT_BLUE,
                "hover_bg": "#357abd",
                "hover_border": "#357abd",
                "pressed_bg": "#2868a3",
                "pressed_border": "#2868a3",
                "checked_bg": MedicalColorPalette.ACCENT_ORANGE,
                "checked_border": MedicalColorPalette.ACCENT_ORANGE,
                "checked_text": "white",
                "disabled_bg": MedicalColorPalette.BACKGROUND_MEDIUM,
                "disabled_text": MedicalColorPalette.TEXT_DISABLED,
                "disabled_border": MedicalColorPalette.BORDER_SUBTLE
            },
            "danger": {
                "bg_color": MedicalColorPalette.ACCENT_RED,
                "text_color": "white",
                "border_color": MedicalColorPalette.ACCENT_RED,
                "hover_bg": "#FF6B35",
                "hover_border": "#FF6B35",
                "pressed_bg": "#CC0000",
                "pressed_border": "#CC0000",
                "checked_bg": "#CC0000",
                "checked_border": "#CC0000",
                "checked_text": "white",
                "disabled_bg": MedicalColorPalette.BACKGROUND_MEDIUM,
                "disabled_text": MedicalColorPalette.TEXT_DISABLED,
                "disabled_border": MedicalColorPalette.BORDER_SUBTLE
            },
            "success": {
                "bg_color": MedicalColorPalette.ACCENT_GREEN,
                "text_color": "white",
                "border_color": MedicalColorPalette.ACCENT_GREEN,
                "hover_bg": "#66CC66",
                "hover_border": "#66CC66",
                "pressed_bg": "#4CAF50",
                "pressed_border": "#4CAF50",
                "checked_bg": "#4CAF50",
                "checked_border": "#4CAF50",
                "checked_text": "white",
                "disabled_bg": MedicalColorPalette.BACKGROUND_MEDIUM,
                "disabled_text": MedicalColorPalette.TEXT_DISABLED,
                "disabled_border": MedicalColorPalette.BORDER_SUBTLE
            }
        }
        
        size_config = size_configs.get(size, size_configs["normal"])
        type_config = type_configs.get(button_type, type_configs["default"])        
        style_params = {**size_config, **type_config}
        
        try:
            return base_style.format(**style_params)
        except KeyError as e:
            self._logger.error(f"Missing style parameter: {e}")
            self._logger.debug(f"Available parameters: {list(style_params.keys())}")
            self._logger.debug(f"Template requiring parameters: {base_style}")
            return """
                QPushButton {
                    background-color: #f0f0f0;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: 500;
                }
            """
    
    def get_medical_panel_style(
        self, 
        panel_type: str = "default",
        has_border: bool = True
    ) -> str:
        panel_configs = {
            "default": {
                "bg_color": MedicalColorPalette.BACKGROUND_MEDIUM,
                "border_color": MedicalColorPalette.BORDER_NORMAL,
                "border_radius": "6px"
            },
            "sidebar": {
                "bg_color": "#f8f9fa",
                "border_color": MedicalColorPalette.BORDER_SUBTLE,
                "border_radius": "0px 8px 8px 0px"
            },
            "dialog": {
                "bg_color": MedicalColorPalette.BACKGROUND_LIGHT,
                "border_color": MedicalColorPalette.BORDER_ACTIVE,
                "border_radius": "8px"
            },
            "group": {
                "bg_color": MedicalColorPalette.BACKGROUND_MEDIUM,
                "border_color": MedicalColorPalette.BORDER_NORMAL,
                "border_radius": "3px"
            }
        }
        
        config = panel_configs.get(panel_type, panel_configs["default"])
        border_style = f"1px solid {config['border_color']}" if has_border else "none"
        
        return f"""
            background-color: {config['bg_color']};
            border: {border_style};
            border-radius: {config['border_radius']};
        """
    
    def get_medical_input_style(self, input_type: str = "default") -> str:
        return f"""
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
                border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
                border-radius: 3px;
                padding: 4px 8px;
                color: {MedicalColorPalette.TEXT_PRIMARY};
                min-height: 20px;
            }}
            
            QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border-color: {MedicalColorPalette.BORDER_ACTIVE};
            }}
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {MedicalColorPalette.ACCENT_BLUE};
                background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
            }}
        """
    
    def get_medical_list_style(self) -> str:
        return f"""
            QListWidget, QTreeView, QTableWidget {{
                background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
                border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
                color: {MedicalColorPalette.TEXT_PRIMARY};
                selection-background-color: {MedicalColorPalette.ACCENT_BLUE};
                alternate-background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
                border-radius: 3px;
            }}
            
            QListWidget::item:hover, QTreeView::item:hover, QTableWidget::item:hover {{
                background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            }}
            
            QListWidget::item:selected, QTreeView::item:selected, QTableWidget::item:selected {{
                background-color: {MedicalColorPalette.ACCENT_BLUE};
                color: white;
            }}
        """
    
    def get_collapsible_sidebar_style(self) -> str:
        return """
            CollapsibleSidebar {
                background-color: #f8f9fa;
                border: none;
                border-radius: 0px 8px 8px 0px;
            }
            
            QWidget#icons-area {
                background-color: #ffffff;
                border-radius: 6px;
                border: none;
            }
            
            QWidget#expanded-area {
                background-color: #ffffff;
                border-radius: 6px;
                border: none;
            }
            
            QPushButton.menu-item {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffffff, stop: 1 #f8f9fa);
                color: #2c3e50;
                border: none;
                text-align: left;
                padding-left: 12px;
                padding-right: 8px;
                font-size: 13px;
                font-weight: 500;
                height: 44px;
                min-height: 44px;
                border-radius: 6px;
                margin: 2px 0px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            
            QPushButton.menu-item:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ebf3fd, stop: 1 #dbeafe);
                border-color: #4a90e2;
                color: #1e3a8a;
                box-shadow: 0 3px 6px rgba(74, 144, 226, 0.2);
                margin-top: -1px;
            }
            
            QPushButton.menu-item:checked {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a90e2, stop: 1 #357abd);
                color: white;
                border-color: #357abd;
                font-weight: 600;
                box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
            }
            
            QPushButton.icon-button {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffffff, stop: 1 #f8f9fa);
                color: #2c3e50;
                border: none;
                border-radius: 6px;
                font-size: 18px;
                font-weight: 500;
                width: 40px;
                height: 40px;
                margin: 1px;
                text-align: center;
                padding: 0px;
            }
            
            QPushButton.icon-button:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ebf3fd, stop: 1 #dbeafe);
                border-color: #4a90e2;
                color: #1e3a8a;
                margin-top: -1px;
            }
            
            QPushButton.icon-button:checked {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a90e2, stop: 1 #357abd);
                color: white;
                border-color: #357abd;
                font-weight: 600;
            }
            
            QPushButton.panel-header {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a90e2, stop: 1 #357abd);
                color: white;
                border: none;
                border-bottom: 1px solid #357abd;
                border-radius: 6px 6px 0px 0px;
                font-size: 13px;
                font-weight: 600;
                text-align: left;
                padding-left: 12px;
                padding-right: 8px;
            }
            
            QPushButton.panel-header:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #357abd, stop: 1 #2868a3);
            }
        """
    
    def get_delete_button_style(self, size: str = "small") -> str:
        size_configs = {
            "small": {"width": "20px", "height": "20px", "font_size": "12px", "border_radius": "10px"},
            "medium": {"width": "28px", "height": "28px", "font_size": "14px", "border_radius": "14px"},
            "large": {"width": "36px", "height": "36px", "font_size": "16px", "border_radius": "18px"}
        }
        
        config = size_configs.get(size, size_configs["small"])
        
        return f"""
            QPushButton {{
                background-color: {MedicalColorPalette.ACCENT_RED};
                color: white;
                border: none;
                border-radius: {config['border_radius']};
                font-weight: bold;
                font-size: {config['font_size']};
                width: {config['width']};
                height: {config['height']};
                max-width: {config['width']};
                max-height: {config['height']};
            }}
            QPushButton:hover {{
                background-color: {MedicalColorPalette.ACCENT_ORANGE};
            }}
            QPushButton:pressed {{
                background-color: #CC0000;
            }}
        """
    
    def get_progress_bar_style(self, color: str = "blue") -> str:
        color_map = {
            "blue": MedicalColorPalette.ACCENT_BLUE,
            "green": MedicalColorPalette.ACCENT_GREEN,
            "orange": MedicalColorPalette.ACCENT_ORANGE,
            "red": MedicalColorPalette.ACCENT_RED
        }
        
        accent_color = color_map.get(color, MedicalColorPalette.ACCENT_BLUE)
        
        return f"""
            QProgressBar {{
                background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
                border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
                border-radius: 3px;
                text-align: center;
                color: {MedicalColorPalette.TEXT_PRIMARY};
                min-height: 20px;
            }}
            
            QProgressBar::chunk {{
                background-color: {accent_color};
                border-radius: 2px;
            }}
        """
    
    def _initialize_component_styles(self) -> Dict[str, Dict[str, str]]:
        return {
            ComponentType.DELETE_BUTTON.value: {
                StyleVariant.DEFAULT.value: self.get_delete_button_style("small"),
                StyleVariant.COMPACT.value: self.get_delete_button_style("small"),
                StyleVariant.LARGE.value: self.get_delete_button_style("large")
            },
            ComponentType.SIDEBAR.value: {
                StyleVariant.DEFAULT.value: self.get_collapsible_sidebar_style()
            },
            ComponentType.PANEL.value: {
                StyleVariant.DEFAULT.value: self.get_medical_panel_style("default"),
                StyleVariant.MEDICAL.value: self.get_medical_panel_style("group")
            },
            ComponentType.INPUT_FIELD.value: {
                StyleVariant.DEFAULT.value: self.get_medical_input_style()
            },
            ComponentType.LIST_WIDGET.value: {
                StyleVariant.DEFAULT.value: self.get_medical_list_style()
            },
            ComponentType.PROGRESS_BAR.value: {
                StyleVariant.DEFAULT.value: self.get_progress_bar_style("blue"),
                StyleVariant.SUCCESS.value: self.get_progress_bar_style("green"),
                StyleVariant.WARNING.value: self.get_progress_bar_style("orange"),
                StyleVariant.DANGER.value: self.get_progress_bar_style("red")
            }
        }
    
    def _build_component_style(
        self, 
        component_type: ComponentType, 
        variant: StyleVariant
    ) -> str:
        component_styles = self._component_styles.get(component_type.value, {})
        return component_styles.get(variant.value, "")
    
    def _apply_custom_properties(self, base_style: str, properties: Dict[str, str]) -> str:
        custom_css = "\n".join(f"{prop}: {value};" for prop, value in properties.items())
        return f"{base_style}\n/* Custom Properties */\n{custom_css}"
    
    def create_medical_dialog_style(self) -> str:
        return f"""
            QDialog {{
                background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
                border: 2px solid {MedicalColorPalette.BORDER_ACTIVE};
                border-radius: 8px;
            }}
            
            QDialogButtonBox {{
                background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
                border-top: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            }}
        """
    
    def create_medical_tooltip_style(self) -> str:
        return f"""
            QToolTip {{
                background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
                border: 1px solid {MedicalColorPalette.BORDER_ACTIVE};
                color: {MedicalColorPalette.TEXT_PRIMARY};
                padding: 4px;
                border-radius: 3px;
                font-size: 11px;
            }}
        """

theme_service = ThemeService()