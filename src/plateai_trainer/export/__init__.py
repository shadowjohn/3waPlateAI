"""ONNX export for v1 crop-recognition and full detector bundles."""
from .bundle import ExportRequest, export_crop_bundle
__all__ = ["ExportRequest", "export_crop_bundle", "DetectionExportRequest", "export_full_bundle"]


def __getattr__(name):
    if name in {"DetectionExportRequest", "export_full_bundle"}:
        from plateai_trainer.detection import export
        return getattr(export, name)
    raise AttributeError(name)
