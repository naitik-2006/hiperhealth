"""
title: Module for MedVision image extraction and analysis.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Union, Optional

from hiperhealth.vision.preprocessing import ImageProcessor, ImageValidationError
from hiperhealth.vision.pipeline import MedVisionPipeline, VisionExtractionResult
from hiperhealth.llm import LLMSettings

FileInput = Union[str, Path, io.BytesIO, bytes]

class MedicalVisionExtractorError(Exception):
    """Base exception for MedVision Extraction."""
    pass

class MedicalVisionExtractor:
    """
    Extracts structured clinical visual observations from images.
    Combines preprocessing (validation, blur check, resize) with Vision LLMs.
    """
    
    def __init__(self, llm_settings: Optional[LLMSettings] = None) -> None:
        self.processor = ImageProcessor()
        self.pipeline = MedVisionPipeline(settings=llm_settings)
        
    def extract_visual_features(
        self, 
        source: FileInput, 
        clinical_context: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Validate, standardize, and analyze the image source.
        """
        try:
            if isinstance(source, bytes):
                file_bytes = source
            elif isinstance(source, io.BytesIO):
                file_bytes = source.read()
                source.seek(0)
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Image not found: {source}")
                file_bytes = path.read_bytes()
            else:
                raise ValueError("Unsupported source type for image analysis")

            self.processor.validate_integrity(file_bytes)
            
            standardized_img = self.processor.standardize(file_bytes)
            
            result: VisionExtractionResult = self.pipeline.analyze_image(
                standardized_img, 
                clinical_context=clinical_context
            )            
            return result.model_dump()
            
        except ImageValidationError as e:
            raise MedicalVisionExtractorError(f"Image preprocessing failed: {e}") from e
        except Exception as e:
            raise MedicalVisionExtractorError(f"Vision analysis failed: {e}") from e

def get_medical_vision_extractor() -> MedicalVisionExtractor:
    """Create and return an instance of MedicalVisionExtractor."""
    return MedicalVisionExtractor()
