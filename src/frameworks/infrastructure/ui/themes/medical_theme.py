from typing import Dict, Any, Tuple
import qdarktheme

class MedicalColorPalette:
    BACKGROUND_DARK = "#0A0A0A"      
    BACKGROUND_MEDIUM = "#1A1A1A"     
    BACKGROUND_LIGHT = "#2A2A2A"      

    TEXT_PRIMARY = "#E8E8E8"          
    TEXT_SECONDARY = "#CCCCCC"        
    TEXT_DISABLED = "#888888"       
    
    ACCENT_BLUE = "#4A90E2"          
    ACCENT_GREEN = "#7ED321"         
    ACCENT_ORANGE = "#FF6B35"      
    ACCENT_RED = "#D0021B"        
    ACCENT_YELLOW = "#F5A623"       
    
    BORDER_SUBTLE = "#404040"         
    BORDER_NORMAL = "#606060"         
    BORDER_ACTIVE = "#4A90E2"        
    
    IMAGE_BLACK = "#000000"            
    IMAGE_WHITE = "#FFFFFF"          
    
    WINDOW_LEVEL_BONE = "#FFFFFF"     
    WINDOW_LEVEL_SOFT = "#CCCCCC"    
    WINDOW_LEVEL_LUNG = "#888888"    

class MedicalThemeManager:
    @staticmethod
    def get_radiology_dark_theme() -> str:
        base_theme = qdarktheme.load_stylesheet(theme="dark")
        
        medical_customizations = f"""
        /* Medical Workstation Theme - Radiology Optimized */
        
        /* Main application background - ultra-dark for eye comfort */
        QMainWindow {{
            background-color: {MedicalColorPalette.BACKGROUND_DARK};
            color: {MedicalColorPalette.TEXT_PRIMARY};
        }}
        
        /* Medical panels and group boxes */
        QGroupBox {{
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 3px;
            margin: 8px 0px;
            padding: 10px 5px 5px 5px;
            font-weight: bold;
            color: {MedicalColorPalette.TEXT_PRIMARY};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 8px 0 8px;
            color: {MedicalColorPalette.ACCENT_BLUE};
            font-weight: bold;
        }}
        
        /* Medical buttons - optimized for precise clicking */
        QPushButton {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 4px;
            padding: 6px 12px;
            color: {MedicalColorPalette.TEXT_PRIMARY};
            font-weight: 500;
            min-height: 20px;
        }}
        
        QPushButton:hover {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
            border-color: {MedicalColorPalette.BORDER_ACTIVE};
        }}
        
        QPushButton:pressed {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
            border-color: {MedicalColorPalette.BORDER_ACTIVE};
        }}
        
        QPushButton:checked {{
            background-color: {MedicalColorPalette.ACCENT_ORANGE};
            border-color: {MedicalColorPalette.ACCENT_ORANGE};
            color: white;
        }}
        
        /* Medical sliders - for Window/Level adjustments */
        QSlider::groove:horizontal {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            height: 6px;
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
            border: 1px solid {MedicalColorPalette.BORDER_ACTIVE};
            width: 14px;
            height: 14px;
            border-radius: 7px;
            margin: -4px 0;
        }}
        
        QSlider::handle:horizontal:hover {{
            background-color: {MedicalColorPalette.ACCENT_ORANGE};
        }}
        
        /* Medical combo boxes and inputs */
        QComboBox {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 3px;
            padding: 4px 8px;
            color: {MedicalColorPalette.TEXT_PRIMARY};
            min-height: 20px;
        }}
        
        QComboBox:hover {{
            border-color: {MedicalColorPalette.BORDER_ACTIVE};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {MedicalColorPalette.TEXT_SECONDARY};
        }}
        
        /* Medical tree views and lists */
        QTreeView, QListView {{
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            color: {MedicalColorPalette.TEXT_PRIMARY};
            selection-background-color: {MedicalColorPalette.ACCENT_BLUE};
            alternate-background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
        }}
        
        QTreeView::item:hover, QListView::item:hover {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
        }}
        
        QTreeView::item:selected, QListView::item:selected {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
            color: white;
        }}
        
        /* Medical tabs - for different imaging modalities */
        QTabWidget::pane {{
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
        }}
        
        QTabBar::tab {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            padding: 8px 16px;
            color: {MedicalColorPalette.TEXT_SECONDARY};
        }}
        
        QTabBar::tab:selected {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
            color: white;
            border-bottom: 2px solid {MedicalColorPalette.ACCENT_BLUE};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            color: {MedicalColorPalette.TEXT_PRIMARY};
        }}
        
        /* Medical status bar */
        QStatusBar {{
            background-color: {MedicalColorPalette.BACKGROUND_DARK};
            border-top: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            color: {MedicalColorPalette.TEXT_SECONDARY};
        }}
        
        /* Medical menu bar */
        QMenuBar {{
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
            border-bottom: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            color: {MedicalColorPalette.TEXT_PRIMARY};
        }}
        
        QMenuBar::item:selected {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
        }}
        
        QMenu {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            color: {MedicalColorPalette.TEXT_PRIMARY};
        }}
        
        QMenu::item:selected {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
        }}
        
        /* Medical scroll bars */
        QScrollBar:vertical {{
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
            width: 14px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 6px;
            margin: 2px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
        }}
        
        /* Medical progress bars */
        QProgressBar {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 3px;
            text-align: center;
            color: {MedicalColorPalette.TEXT_PRIMARY};
        }}
        
        QProgressBar::chunk {{
            background-color: {MedicalColorPalette.ACCENT_BLUE};
            border-radius: 2px;
        }}
        
        /* Medical tooltips */
        QToolTip {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_ACTIVE};
            color: {MedicalColorPalette.TEXT_PRIMARY};
            padding: 4px;
            border-radius: 3px;
        }}
        
        /* Medical image displays - pure black for DICOM compliance */
        QWidget[objectName="MedicalImageCanvas"] {{
            background-color: {MedicalColorPalette.IMAGE_BLACK};
            border: 1px solid {MedicalColorPalette.BORDER_SUBTLE};
        }}
        
        QWidget[objectName="medical_image_display"] {{
            background-color: {MedicalColorPalette.IMAGE_BLACK};
            border: 1px solid {MedicalColorPalette.BORDER_SUBTLE};
        }}
        
        /* VTK rendering widgets for 3D displays */
        QVTKOpenGLNativeWidget {{
            background-color: {MedicalColorPalette.IMAGE_BLACK};
            border: 1px solid {MedicalColorPalette.BORDER_SUBTLE};
        }}
        
        /* Splitter handles - invisible/minimal styling */
        QSplitter::handle {{
            background-color: transparent;
            border: none;
        }}
        
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        
        QSplitter::handle:vertical {{
            height: 1px;
        }}
        
        /* Medical spin boxes and input fields */
        QSpinBox, QDoubleSpinBox {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 3px;
            padding: 4px 8px;
            color: {MedicalColorPalette.TEXT_PRIMARY};
            min-height: 20px;
        }}
        
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {MedicalColorPalette.BORDER_ACTIVE};
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {MedicalColorPalette.ACCENT_BLUE};
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
        }}
        
        /* Line edits for medical data entry */
        QLineEdit {{
            background-color: {MedicalColorPalette.BACKGROUND_LIGHT};
            border: 1px solid {MedicalColorPalette.BORDER_NORMAL};
            border-radius: 3px;
            padding: 4px 8px;
            color: {MedicalColorPalette.TEXT_PRIMARY};
            min-height: 20px;
        }}
        
        QLineEdit:hover {{
            border-color: {MedicalColorPalette.BORDER_ACTIVE};
        }}
        
        QLineEdit:focus {{
            border-color: {MedicalColorPalette.ACCENT_BLUE};
            background-color: {MedicalColorPalette.BACKGROUND_MEDIUM};
        }}
        """
        
        return base_theme + medical_customizations
    
    @staticmethod
    def get_dicom_display_function() -> Dict[str, float]:
        """
        Returns DICOM Grayscale Standard Display Function parameters.
        
        Based on DICOM Part 14: Grayscale Standard Display Function (GSDF).
        This ensures consistent luminance characteristics across different
        display systems for diagnostic accuracy.
        
        Returns:
            Dictionary of DICOM GSDF parameters
        """
        return {
            "min_luminance": 1.0,      
            "max_luminance": 4000.0,  
            "gamma": 2.2,             
            "ambient_luminance": 1.0, 
            "just_noticeable_difference": 1.0, 
            "bit_depth": 8,         
            "contrast_sensitivity": 3.4  
        }
    
    @staticmethod
    def get_medical_color_maps() -> Dict[str, Dict[str, str]]:
        return {
            "ct_bone": {
                "name": "CT Bone",
                "window": "1500",
                "level": "300",
                "description": "Optimized for bone visualization"
            },
            "ct_soft_tissue": {
                "name": "CT Soft Tissue", 
                "window": "400",
                "level": "40",
                "description": "Optimized for soft tissue contrast"
            },
            "ct_lung": {
                "name": "CT Lung",
                "window": "1600", 
                "level": "-600",
                "description": "Optimized for lung parenchyma"
            },
            "mri_t1": {
                "name": "MRI T1",
                "window": "600",
                "level": "300",
                "description": "T1-weighted imaging contrast"
            },
            "mri_t2": {
                "name": "MRI T2",
                "window": "800",
                "level": "400", 
                "description": "T2-weighted imaging contrast"
            },
            "xray": {
                "name": "X-Ray",
                "window": "2000",
                "level": "1000",
                "description": "High contrast radiography"
            },
            "mammography": {
                "name": "Mammography",
                "window": "4000",
                "level": "2000",
                "description": "Breast tissue imaging"
            },
            "pet_ct": {
                "name": "PET/CT",
                "window": "400",
                "level": "200",
                "description": "Combined metabolic and anatomical imaging"
            },
            "ultrasound": {
                "name": "Ultrasound",
                "window": "255",
                "level": "128",
                "description": "Real-time tissue imaging"
            }
        }
    
    @staticmethod
    def get_annotation_colors() -> Dict[str, str]:
        return {
            "measurement": MedicalColorPalette.ACCENT_YELLOW,     
            "roi_normal": MedicalColorPalette.ACCENT_GREEN,       
            "roi_abnormal": MedicalColorPalette.ACCENT_RED,       
            "roi_suspicious": MedicalColorPalette.ACCENT_ORANGE, 
            "annotation": MedicalColorPalette.ACCENT_BLUE,        
            "reference": MedicalColorPalette.TEXT_SECONDARY,     
        }
    
    @staticmethod
    def get_pathology_color_scheme() -> Dict[str, str]:
        return {
            "benign": "#7ED321",         
            "likely_benign": "#9FD356",  
            "indeterminate": "#F5A623",    
            "suspicious": "#FF6B35",      
            "malignant": "#D0021B",        

            "pirads_1": "#00FF00",         
            "pirads_2": "#80FF00",       
            "pirads_3": "#FFFF00",        
            "pirads_4": "#FF8000",    
            "pirads_5": "#FF0000",      
            
            "healthy_tissue": "#66CC66", 
            "inflammation": "#FFCC00",    
            "fibrosis": "#CC9966",         
            "necrosis": "#666666",      
            "hemorrhage": "#CC0000",     
            "calcification": "#FFFFFF",   
            
            "artery": "#FF6666",          
            "vein": "#6666FF",            
            "capillary": "#FF99FF",      
            
            "peripheral_zone": "#4A90E2", 
            "transition_zone": "#9013FE",  
            "central_zone": "#00BCD4",    
            "anterior_zone": "#795548",   
        }
    
    @staticmethod
    def get_contrast_enhancement_colors() -> Dict[str, str]:
        return {
            "no_enhancement": "#000000",     
            "mild_enhancement": "#404040",  
            "moderate_enhancement": "#808080", 
            "strong_enhancement": "#C0C0C0",  
            "avid_enhancement": "#FFFFFF",   
            
            "washout_pattern": "#FF4444",   
            "persistent_pattern": "#44FF44", 
            "plateau_pattern": "#FFFF44",    
        }

def create_medical_radiology_theme() -> str:
    return MedicalThemeManager.get_radiology_dark_theme()