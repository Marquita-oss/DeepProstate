import logging
import json
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from deepprostate.core.domain.entities.ai_analysis import AIAnalysisResult
from deepprostate.core.domain.entities.segmentation import MedicalSegmentation


class AIAnalysisExportService:
    def __init__(self, export_base_path: Path = None):
        self._logger = logging.getLogger(__name__)

        self._export_base_path = export_base_path or Path("./ai_analysis_exports")
        self._export_base_path.mkdir(exist_ok=True)

        self._supported_mask_formats = ["nifti", "dicom_seg", "stl", "numpy"]
        self._supported_report_formats = ["pdf", "html", "json"]

        self._logger.info("AI Analysis Export Service initialized")

    def export_analysis_masks(self,
                             analysis_result: AIAnalysisResult,
                             format: str = "nifti",
                             export_path: Optional[Path] = None) -> List[Path]:
        try:
            if format not in self._supported_mask_formats:
                raise ValueError(f"Unsupported format: {format}. Supported: {self._supported_mask_formats}")

            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = self._export_base_path / f"masks_{analysis_result.analysis_id}_{timestamp}"
            else:
                export_dir = export_path

            export_dir.mkdir(parents=True, exist_ok=True)

            exported_files = []

            for i, overlay_data in enumerate(analysis_result.overlay_data):
                filename = f"{overlay_data.anatomical_region.value}_{i:02d}"

                if format == "nifti":
                    file_path = export_dir / f"{filename}.nii.gz"
                    self._export_mask_as_nifti(overlay_data.mask_array, file_path, overlay_data)
                    exported_files.append(file_path)

                elif format == "numpy":
                    file_path = export_dir / f"{filename}.npy"
                    np.save(file_path, overlay_data.mask_array)
                    exported_files.append(file_path)

                elif format == "dicom_seg":
                    file_path = export_dir / f"{filename}.dcm"
                    self._export_mask_as_dicom_seg(overlay_data.mask_array, file_path, overlay_data, analysis_result)
                    exported_files.append(file_path)

                elif format == "stl":
                    file_path = export_dir / f"{filename}.stl"
                    self._export_mask_as_stl(overlay_data.mask_array, file_path, overlay_data)
                    exported_files.append(file_path)

            self._logger.info(f"Exported {len(exported_files)} masks in {format} format to {export_dir}")
            return exported_files

        except Exception as e:
            self._logger.error(f"Failed to export masks: {e}")
            return []

    def export_analysis_report(self,
                              analysis_result: AIAnalysisResult,
                              format: str = "html",
                              export_path: Optional[Path] = None) -> Optional[Path]:
        try:
            if format not in self._supported_report_formats:
                raise ValueError(f"Unsupported format: {format}. Supported: {self._supported_report_formats}")

            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ai_analysis_report_{analysis_result.analysis_id}_{timestamp}.{format}"
                export_path = self._export_base_path / filename

            report_data = self._generate_report_data(analysis_result)

            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)

            elif format == "html":
                html_content = self._generate_html_report(report_data)
                with open(export_path, 'w') as f:
                    f.write(html_content)

            elif format == "pdf":
                html_path = export_path.with_suffix('.html')
                html_content = self._generate_html_report(report_data)
                with open(html_path, 'w') as f:
                    f.write(html_content)
                export_path = html_path
                self._logger.warning("PDF generation not implemented, exported as HTML")

            self._logger.info(f"Exported analysis report to {export_path}")
            return export_path

        except Exception as e:
            self._logger.error(f"Failed to export report: {e}")
            return None

    def export_batch_results(self,
                           analysis_results: List[AIAnalysisResult],
                           export_path: Optional[Path] = None) -> Optional[Path]:
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = self._export_base_path / f"batch_export_{timestamp}"
            else:
                export_dir = export_path

            export_dir.mkdir(parents=True, exist_ok=True)

            for result in analysis_results:
                analysis_dir = export_dir / f"analysis_{result.analysis_id}"
                analysis_dir.mkdir(exist_ok=True)

                self.export_analysis_masks(result, "nifti", analysis_dir / "masks")

                self.export_analysis_report(result, "json", analysis_dir / "report.json")

            batch_summary = self._generate_batch_summary(analysis_results)
            with open(export_dir / "batch_summary.json", 'w') as f:
                json.dump(batch_summary, f, indent=2, default=str)

            self._logger.info(f"Exported batch of {len(analysis_results)} analyses to {export_dir}")
            return export_dir

        except Exception as e:
            self._logger.error(f"Failed to export batch results: {e}")
            return None

    def get_export_formats(self) -> Dict[str, List[str]]:
        return {
            "masks": self._supported_mask_formats,
            "reports": self._supported_report_formats
        }

    def _export_mask_as_nifti(self, mask_array: np.ndarray, file_path: Path, overlay_data) -> None:
        try:
            import nibabel as nib

            nifti_img = nib.Nifti1Image(mask_array.astype(np.uint8), affine=np.eye(4))

            nifti_img.header['descrip'] = f"AI Analysis - {overlay_data.anatomical_region.value}".encode()

            nib.save(nifti_img, str(file_path))

        except ImportError:
            np.savez(file_path.with_suffix('.npz'),
                    mask=mask_array,
                    anatomical_region=overlay_data.anatomical_region.value,
                    confidence_score=overlay_data.confidence_score)
            self._logger.warning(f"NiBabel not available, saved as .npz: {file_path}")

    def _export_mask_as_dicom_seg(self, mask_array: np.ndarray, file_path: Path, overlay_data, analysis_result) -> None:
        self._logger.warning("DICOM-SEG export not yet implemented")
        np.save(file_path.with_suffix('.npy'), mask_array)

    def _export_mask_as_stl(self, mask_array: np.ndarray, file_path: Path, overlay_data) -> None:
        try:
            from skimage import measure

            verts, faces, normals, values = measure.marching_cubes(mask_array.astype(np.uint8), level=0.5)

            with open(file_path, 'w') as f:
                f.write(f"solid {overlay_data.anatomical_region.value}\n")
                for i in range(0, len(faces)):
                    face = faces[i]
                    normal = normals[i]
                    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("    outer loop\n")
                    for j in range(3):
                        vertex = verts[face[j]]
                        f.write(f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                f.write(f"endsolid {overlay_data.anatomical_region.value}\n")

        except ImportError:
            self._logger.warning("scikit-image not available for STL export")
            np.save(file_path.with_suffix('.npy'), mask_array)

    def _generate_report_data(self, analysis_result: AIAnalysisResult) -> Dict[str, Any]:
        return {
            "analysis_info": {
                "analysis_id": analysis_result.analysis_id,
                "analysis_type": analysis_result.analysis_type.value,
                "created_timestamp": analysis_result.created_timestamp.isoformat() if analysis_result.created_timestamp else None,
                "processing_time_ms": analysis_result.processing_metadata.get("processing_time_ms", 0)
            },
            "patient_context": analysis_result.patient_context,
            "segmentations": [
                {
                    "anatomical_region": seg.anatomical_region.value,
                    "volume_voxels": seg.metrics.volume_voxels if seg.metrics else 0,
                    "confidence_level": seg.confidence_level.value if seg.confidence_level else "unknown"
                }
                for seg in analysis_result.segmentations
            ],
            "overlays": [
                {
                    "anatomical_region": overlay.anatomical_region.value,
                    "confidence_score": overlay.confidence_score,
                    "volume_mm3": overlay.volume_mm3,
                    "mask_shape": list(overlay.mask_array.shape)
                }
                for overlay in analysis_result.overlay_data
            ],
            "processing_metadata": analysis_result.processing_metadata,
            "export_timestamp": datetime.now().isoformat()
        }

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Analysis Report - {report_data['analysis_info']['analysis_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Analysis Report</h1>
                <p>Analysis ID: {report_data['analysis_info']['analysis_id']}</p>
                <p>Type: {report_data['analysis_info']['analysis_type']}</p>
                <p>Generated: {report_data['export_timestamp']}</p>
            </div>

            <div class="section">
                <h2>Patient Information</h2>
                <table>
                    <tr><th>Patient ID</th><td>{report_data['patient_context'].get('patient_id', 'N/A')}</td></tr>
                    <tr><th>Series Description</th><td>{report_data['patient_context'].get('series_description', 'N/A')}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Analysis Results</h2>
                <table>
                    <tr><th>Anatomical Region</th><th>Confidence Score</th><th>Volume (mm³)</th></tr>
        """

        for overlay in report_data['overlays']:
            html += f"""
                    <tr>
                        <td>{overlay['anatomical_region']}</td>
                        <td>{overlay['confidence_score']:.3f}</td>
                        <td>{overlay['volume_mm3']:.1f}</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="section">
                <h2>Processing Information</h2>
                <table>
                    <tr><th>Processing Time</th><td>{:.1f} seconds</td></tr>
                </table>
            </div>
        </body>
        </html>
        """.format(report_data['analysis_info']['processing_time_ms'] / 1000.0)

        return html

    def _generate_batch_summary(self, analysis_results: List[AIAnalysisResult]) -> Dict[str, Any]:
        return {
            "batch_info": {
                "total_analyses": len(analysis_results),
                "export_timestamp": datetime.now().isoformat(),
                "analysis_types": list(set(result.analysis_type.value for result in analysis_results))
            },
            "analyses": [
                {
                    "analysis_id": result.analysis_id,
                    "analysis_type": result.analysis_type.value,
                    "patient_id": result.patient_context.get("patient_id", "Unknown"),
                    "segmentation_count": len(result.segmentations)
                }
                for result in analysis_results
            ]
        }