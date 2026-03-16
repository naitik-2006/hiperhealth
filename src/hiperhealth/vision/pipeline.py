from __future__ import annotations

import base64
import io

from PIL import Image
from pydantic import BaseModel
from hiperhealth.llm import LLMSettings, build_structured_llm

class VisionFinding(BaseModel):
    """Structured output from the Vision LLM."""

    region_tag: str
    clinical_finding: str
    confidence_score: float

class VisionExtractionResult(BaseModel):
    """Collection of findings from a single image."""

    findings: list[VisionFinding]
    image_quality_notes: str

class MedVisionPipeline:
    """Handles communication with Vision models."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        """Initialize with LLMSettings (defaults to OpenAI o4-mini/gpt-4o-mini)."""
        self.settings = settings or LLMSettings(
            provider='openai', model='gpt-4o-mini'
        )
        self.llm = build_structured_llm(settings=self.settings)

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{img_str}'

    def analyze_image(
        self, processed_image: Image.Image, clinical_context: str | None = None
    ) -> VisionExtractionResult:
        """
        Send standardized image to the vision model and extract structured findings.
        """

        base64_img = self._image_to_base64(processed_image)
        schema_json = VisionExtractionResult.model_json_schema()

        system_prompt = (
            "You are an expert medical computer vision assistant. "
            "Analyze the provided clinical image for symptoms in eyes, nails, skin, tongue, or ears. "
            "Return a structured list of findings with confidence scores. "
            f"\n\nOutput must strictly follow this JSON schema:\n{schema_json}"
        )

        user_prompt = "Please analyze this clinical image."
        if clinical_context:
            user_prompt += f" Context: {clinical_context}"

        from litellm import completion
        from hiperhealth.llm import _coerce_model_output, _extract_message_content

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": base64_img}},
                ],
            },
        ]

        # Single optimized call
        kwargs = self.settings.to_litellm_kwargs()
        response = completion(
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

        content = _extract_message_content(response)
        return _coerce_model_output(content, VisionExtractionResult)
