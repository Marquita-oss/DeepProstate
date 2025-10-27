"""
infrastructure/ui/widgets/image_information_panel.py

Panel of información completo for imágenes médicas que muestra:
- Patient Demographics 
- Study Information
- Series Information
- Image Technical Data
- DICOM Metadata
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QGroupBox, QGridLayout, QTextEdit, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QFont

from deepprostate.core.domain.entities.medical_image import MedicalImage


class ImageInformationPanel(QWidget):
    """
    Panel completo of información médica que reemplaza la funcionalidad
    of "Show Info" eliminada of the Patient Studies panel.
    """
    
    # Señales for comunicación with otros componentes
    patient_info_updated = pyqtSignal(dict)  # patient_data
    study_info_updated = pyqtSignal(dict)    # study_data
    series_info_updated = pyqtSignal(dict)   # series_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Estado interno
        self._current_image: Optional[MedicalImage] = None
        self._setup_ui()
        self._clear_all_info()
        
        self._logger.info("ImageInformationPanel initialized")
    
    def _setup_ui(self):
        """Configura la interfaz completa of the panel."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Scroll area for manejar contenido largo
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Widget contenedor of the contenido
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(12)
        
        # Título of the panel
        title_label = QLabel("📊 Image Information")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(title_label)
        
        # Secciones of información
        self._create_patient_section(content_layout)
        self._create_study_section(content_layout)
        self._create_series_section(content_layout)
        self._create_image_properties_section(content_layout)
        self._create_dicom_metadata_section(content_layout)
        
        # Espaciador al final
        content_layout.addStretch()
        
        # Configure scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
    
    def sizeHint(self) -> QSize:
        """
        Proporciona el tamaño preferido for the panel of información of imagen.
        Se calcula dinámicamente basado in the contenido real.
        """
        # Ancho mínimo for mostrar labels e información cómodamente
        # Considerando:
        # - Labels más largos como "Series Instance UID:"
        # - UIDs que puedin ser of 50+ caracteres
        # - Márgenes y padding
        preferred_width = 340
        
        # Altura basada in the número of secciones expandidas
        # Cada sección tiene aproximadamente:
        # - Patient: ~120px, Study: ~120px, Series: ~150px, 
        # - Image Properties: ~150px, DICOM Metadata: ~200px
        base_height = 80  # Header y padding
        patient_height = 120
        study_height = 120  
        series_height = 150
        image_height = 150
        metadata_height = 200
        
        total_height = (base_height + patient_height + study_height + 
                       series_height + image_height + metadata_height)
        
        return QSize(preferred_width, total_height)
    
    def minimumSizeHint(self) -> QSize:
        """Tamaño mínimo requerido for mostrar el contenido básico."""
        return QSize(280, 400)  # Mínimo for mostrar al menos 2-3 secciones
    
    def _create_patient_section(self, parent_layout: QVBoxLayout):
        """Crea sección of información of the paciente."""
        self.patient_group = QGroupBox("👤 Patient Demographics")
        self.patient_group.setFlat(False)
        patient_layout = QGridLayout(self.patient_group)
        
        # Labels for información of the paciente
        self.patient_name_label = QLabel("N/A")
        self.patient_id_label = QLabel("N/A") 
        self.patient_birth_label = QLabel("N/A")
        self.patient_sex_label = QLabel("N/A")
        self.patient_age_label = QLabel("N/A")
        
        # Layout of la información
        patient_layout.addWidget(QLabel("Name:"), 0, 0)
        patient_layout.addWidget(self.patient_name_label, 0, 1)
        patient_layout.addWidget(QLabel("ID:"), 1, 0)
        patient_layout.addWidget(self.patient_id_label, 1, 1)
        patient_layout.addWidget(QLabel("Birth Date:"), 2, 0)
        patient_layout.addWidget(self.patient_birth_label, 2, 1)
        patient_layout.addWidget(QLabel("Sex:"), 3, 0)
        patient_layout.addWidget(self.patient_sex_label, 3, 1)
        patient_layout.addWidget(QLabel("Age:"), 4, 0)
        patient_layout.addWidget(self.patient_age_label, 4, 1)
        
        parent_layout.addWidget(self.patient_group)
    
    def _create_study_section(self, parent_layout: QVBoxLayout):
        """Crea sección of información of the estudio."""
        self.study_group = QGroupBox("📁 Study Information")
        study_layout = QGridLayout(self.study_group)
        
        # Labels for información of the estudio
        self.study_description_label = QLabel("N/A")
        self.study_date_label = QLabel("N/A")
        self.study_time_label = QLabel("N/A")
        self.study_uid_label = QLabel("N/A")
        self.referring_physician_label = QLabel("N/A")
        
        # Layout of la información
        study_layout.addWidget(QLabel("Description:"), 0, 0)
        study_layout.addWidget(self.study_description_label, 0, 1)
        study_layout.addWidget(QLabel("Date:"), 1, 0)
        study_layout.addWidget(self.study_date_label, 1, 1)
        study_layout.addWidget(QLabel("Time:"), 2, 0)
        study_layout.addWidget(self.study_time_label, 2, 1)
        study_layout.addWidget(QLabel("Study UID:"), 3, 0)
        study_layout.addWidget(self.study_uid_label, 3, 1)
        study_layout.addWidget(QLabel("Physician:"), 4, 0)
        study_layout.addWidget(self.referring_physician_label, 4, 1)
        
        parent_layout.addWidget(self.study_group)
    
    def _create_series_section(self, parent_layout: QVBoxLayout):
        """Crea sección of información of la serie."""
        self.series_group = QGroupBox("📄 Series Information")
        series_layout = QGridLayout(self.series_group)
        
        # Labels for información of la serie
        self.series_description_label = QLabel("N/A")
        self.series_number_label = QLabel("N/A")
        self.series_uid_label = QLabel("N/A")
        self.modality_label = QLabel("N/A")
        self.sequence_name_label = QLabel("N/A")
        self.body_part_label = QLabel("N/A")
        
        # Layout of la información
        series_layout.addWidget(QLabel("Description:"), 0, 0)
        series_layout.addWidget(self.series_description_label, 0, 1)
        series_layout.addWidget(QLabel("Number:"), 1, 0)
        series_layout.addWidget(self.series_number_label, 1, 1)
        series_layout.addWidget(QLabel("Series UID:"), 2, 0)
        series_layout.addWidget(self.series_uid_label, 2, 1)
        series_layout.addWidget(QLabel("Modality:"), 3, 0)
        series_layout.addWidget(self.modality_label, 3, 1)
        series_layout.addWidget(QLabel("Sequence:"), 4, 0)
        series_layout.addWidget(self.sequence_name_label, 4, 1)
        series_layout.addWidget(QLabel("Body Part:"), 5, 0)
        series_layout.addWidget(self.body_part_label, 5, 1)
        
        parent_layout.addWidget(self.series_group)
    
    def _create_image_properties_section(self, parent_layout: QVBoxLayout):
        """Crea sección of propiedades técnicas of la imagen."""
        self.image_props_group = QGroupBox("🖼️ Image Properties")
        props_layout = QGridLayout(self.image_props_group)
        
        # Labels for propiedades of la imagen
        self.dimensions_label = QLabel("N/A")
        self.spacing_label = QLabel("N/A")
        self.slice_thickness_label = QLabel("N/A")
        self.orientation_label = QLabel("N/A")
        self.data_type_label = QLabel("N/A")
        self.file_size_label = QLabel("N/A")
        
        # Layout of la información
        props_layout.addWidget(QLabel("Dimensions:"), 0, 0)
        props_layout.addWidget(self.dimensions_label, 0, 1)
        props_layout.addWidget(QLabel("Spacing:"), 1, 0)
        props_layout.addWidget(self.spacing_label, 1, 1)
        props_layout.addWidget(QLabel("Slice Thickness:"), 2, 0)
        props_layout.addWidget(self.slice_thickness_label, 2, 1)
        props_layout.addWidget(QLabel("Orientation:"), 3, 0)
        props_layout.addWidget(self.orientation_label, 3, 1)
        props_layout.addWidget(QLabel("Data Type:"), 4, 0)
        props_layout.addWidget(self.data_type_label, 4, 1)
        props_layout.addWidget(QLabel("File Size:"), 5, 0)
        props_layout.addWidget(self.file_size_label, 5, 1)
        
        parent_layout.addWidget(self.image_props_group)
    
    def _create_dicom_metadata_section(self, parent_layout: QVBoxLayout):
        """Crea sección of metadata DICOM completo."""
        self.dicom_group = QGroupBox("🏥 DICOM Metadata")
        dicom_layout = QVBoxLayout(self.dicom_group)
        
        # Text area for metadata completo
        self.dicom_metadata_text = QTextEdit()
        self.dicom_metadata_text.setMaximumHeight(200)
        self.dicom_metadata_text.setReadOnly(True)
        self.dicom_metadata_text.setFont(QFont("Courier", 8))  # Monospace font
        self.dicom_metadata_text.setPlainText("No DICOM metadata available")
        
        dicom_layout.addWidget(self.dicom_metadata_text)
        parent_layout.addWidget(self.dicom_group)
    
    @pyqtSlot(object)
    def update_image_information(self, medical_image: MedicalImage):
        """
        Actualiza toda la información basada in una image médica cargada.
        
        Args:
            medical_image: Imagin médica with metadata completo
        """
        self._logger.debug(f"Updating Image Information panel with medical image: {medical_image}")
        
        if not medical_image:
            self._logger.warning("No medical image provided to Image Information panel")
            self._clear_all_info()
            return
        
        self._current_image = medical_image
        
        try:
            # Update cada sección
            self._update_patient_info(medical_image)
            self._update_study_info(medical_image)
            self._update_series_info(medical_image)
            self._update_image_properties(medical_image)
            self._update_dicom_metadata(medical_image)
            
            self._logger.info(f"Image information updated for: {medical_image.series_instance_uid}")
            
        except Exception as e:
            self._logger.error(f"Error updating image information: {e}")
            self._clear_all_info()
    
    def _update_patient_info(self, medical_image: MedicalImage):
        """Actualiza información of the paciente."""
        # Use public properties and get_dicom_tag method
        patient_name = medical_image.get_dicom_tag('PatientName') or 'Unknown Patient'
        patient_id = medical_image.patient_id  # This is a public property
        birth_date = medical_image.get_dicom_tag('PatientBirthDate') or 'Unknown'
        sex = medical_image.get_dicom_tag('PatientSex') or 'Unknown'
        age = medical_image.get_dicom_tag('PatientAge') or 'Unknown'
        
        # Update UI
        self.patient_name_label.setText(str(patient_name))
        self.patient_id_label.setText(str(patient_id))
        self.patient_birth_label.setText(str(birth_date))
        self.patient_sex_label.setText(str(sex))
        self.patient_age_label.setText(str(age))
    
    def _update_study_info(self, medical_image: MedicalImage):
        """Actualiza información of the estudio."""
        # Use public properties and get_dicom_tag method
        study_description = medical_image.get_dicom_tag('StudyDescription') or 'Unknown Study'
        study_date = medical_image.get_dicom_tag('StudyDate') or medical_image.acquisition_date.strftime('%Y-%m-%d')
        study_time = medical_image.get_dicom_tag('StudyTime') or medical_image.acquisition_date.strftime('%H:%M:%S')
        study_uid = medical_image.study_instance_uid  # This is a public property
        referring_physician = medical_image.get_dicom_tag('ReferringPhysicianName') or 'Unknown'
        
        # Update UI
        self.study_description_label.setText(str(study_description))
        self.study_date_label.setText(str(study_date))
        self.study_time_label.setText(str(study_time))
        self.study_uid_label.setText(str(study_uid)[:50] + "..." if len(str(study_uid)) > 50 else str(study_uid))
        self.referring_physician_label.setText(str(referring_physician))
    
    def _update_series_info(self, medical_image: MedicalImage):
        """Actualiza información of la serie."""
        # Use public properties and get_dicom_tag method
        series_description = medical_image.get_dicom_tag('SeriesDescription') or 'Unknown Series'
        series_number = medical_image.get_dicom_tag('SeriesNumber') or 'Unknown'
        series_uid = medical_image.series_instance_uid  # This is a public property
        modality = medical_image.modality.value if hasattr(medical_image.modality, 'value') else str(medical_image.modality)
        sequence_name = medical_image.get_dicom_tag('SequenceName') or 'Unknown'
        body_part = medical_image.get_dicom_tag('BodyPartExamined') or 'Unknown'
        
        # Update UI
        self.series_description_label.setText(str(series_description))
        self.series_number_label.setText(str(series_number))
        self.series_uid_label.setText(str(series_uid)[:50] + "..." if len(str(series_uid)) > 50 else str(series_uid))
        self.modality_label.setText(str(modality))
        self.sequence_name_label.setText(str(sequence_name))
        self.body_part_label.setText(str(body_part))
    
    def _update_image_properties(self, medical_image: MedicalImage):
        """Actualiza propiedades técnicas of la imagen."""
        if medical_image.image_data is None:
            self._clear_image_properties()
            return
        
        # Extract image properties using public properties
        shape = medical_image.dimensions  # Use public property
        spacing = medical_image.spacing  # Use public property
        
        # Format information
        dimensions = f"{shape[2] if len(shape) > 2 else 1}×{shape[1]}×{shape[0]}"
        spacing_str = f"{spacing.x:.2f}×{spacing.y:.2f}×{spacing.z:.2f} mm"
        slice_thickness = medical_image.get_dicom_tag('SliceThickness') or 'Unknown'
        orientation = medical_image.get_dicom_tag('ImageOrientationPatient') or 'Unknown'
        data_type = str(medical_image.original_data_type)
        
        # Estimate file size
        file_size = medical_image.image_data.nbytes
        if file_size > 1024*1024:
            file_size_str = f"{file_size/(1024*1024):.1f} MB"
        elif file_size > 1024:
            file_size_str = f"{file_size/1024:.1f} KB"
        else:
            file_size_str = f"{file_size} B"
        
        # Update UI
        self.dimensions_label.setText(dimensions)
        self.spacing_label.setText(spacing_str)
        self.slice_thickness_label.setText(f"{slice_thickness} mm" if slice_thickness != 'Unknown' else slice_thickness)
        self.orientation_label.setText(str(orientation)[:20] + "..." if len(str(orientation)) > 20 else str(orientation))
        self.data_type_label.setText(data_type)
        self.file_size_label.setText(file_size_str)
    
    def _update_dicom_metadata(self, medical_image: MedicalImage):
        """Actualiza metadata DICOM completo."""
        metadata = getattr(medical_image, 'dicom_metadata', {})
        
        if not metadata:
            self.dicom_metadata_text.setPlainText("No DICOM metadata available")
            return
        
        # Format metadata for display
        formatted_metadata = []
        for key, value in metadata.items():
            # Limit very long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."
            formatted_metadata.append(f"{key}: {str_value}")
        
        # Sort and join
        formatted_metadata.sort()
        metadata_text = "\n".join(formatted_metadata)
        self.dicom_metadata_text.setPlainText(metadata_text)
    
    def _clear_all_info(self):
        """Limpia toda la información mostrada."""
        self._clear_patient_info()
        self._clear_study_info()
        self._clear_series_info()
        self._clear_image_properties()
        self.dicom_metadata_text.setPlainText("No image loaded")
    
    def _clear_patient_info(self):
        """Limpia información of the paciente."""
        self.patient_name_label.setText("No image loaded")
        self.patient_id_label.setText("N/A")
        self.patient_birth_label.setText("N/A")
        self.patient_sex_label.setText("N/A")
        self.patient_age_label.setText("N/A")
    
    def _clear_study_info(self):
        """Limpia información of the estudio."""
        self.study_description_label.setText("No image loaded")
        self.study_date_label.setText("N/A")
        self.study_time_label.setText("N/A")
        self.study_uid_label.setText("N/A")
        self.referring_physician_label.setText("N/A")
    
    def _clear_series_info(self):
        """Limpia información of la serie."""
        self.series_description_label.setText("No image loaded")
        self.series_number_label.setText("N/A")
        self.series_uid_label.setText("N/A")
        self.modality_label.setText("N/A")
        self.sequence_name_label.setText("N/A")
        self.body_part_label.setText("N/A")
    
    def _clear_image_properties(self):
        """Limpia propiedades of la imagen."""
        self.dimensions_label.setText("N/A")
        self.spacing_label.setText("N/A")
        self.slice_thickness_label.setText("N/A")
        self.orientation_label.setText("N/A")
        self.data_type_label.setText("N/A")
        self.file_size_label.setText("N/A")
    
    def get_current_image(self) -> Optional[MedicalImage]:
        """Retorna la image actualmente mostrada."""
        return self._current_image