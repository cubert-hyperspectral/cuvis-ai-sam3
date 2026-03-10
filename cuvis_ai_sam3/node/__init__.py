"""cuvis_ai_sam3 node definitions.

Node classes are registered via `cuvis_ai_sam3.register_all_nodes()`.
"""

from .prompt_video import SAM3ObjectTracker
from .sam3_streaming_propagation import SAM3StreamingPropagation
from .sam3_video_tracker import SAM3TrackerInference
from .spectral_signature_extractor import SpectralSignatureExtractor

__all__ = [
    "SAM3ObjectTracker",
    "SAM3StreamingPropagation",
    "SAM3TrackerInference",
    "SpectralSignatureExtractor",
]
