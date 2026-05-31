"""
Google MLKit Vision Equivalent - Text Detection & Recognition using MediaPipe
This module provides on-device text detection similar to Google MLKit Vision
without requiring API keys or external services.
"""

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_text_detector = mp.tasks.vision.TextDetector
TextDetectorOptions = mp.tasks.vision.TextDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class GoogleMLKitVisionExtractor:
    """
    Google MLKit Vision equivalent using MediaPipe.
    Provides text detection and layout analysis similar to Google MLKit Text Recognition.
    """
    
    def __init__(self):
        """Initialize MediaPipe text detector"""
        self.detector = None
        self.initialize_detector()
    
    def initialize_detector(self):
        """Initialize MediaPipe text detector"""
        try:
            options = TextDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=None  # Uses default bundled model
                ),
                running_mode=VisionRunningMode.IMAGE
            )
            self.detector = mp_text_detector.create_from_options(options)
            logger.info("✅ MediaPipe Text Detector initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  MediaPipe initialization error: {e}")
            logger.info("Falling back to traditional OCR methods")
            self.detector = None
    
    def detect_text(self, image: np.ndarray) -> Tuple[str, List[dict]]:
        """
        Detect text in image using MediaPipe (Google MLKit Vision equivalent)
        
        Args:
            image: Input image as numpy array (BGR or RGB)
        
        Returns:
            Tuple of (extracted_text, detections_list)
        """
        if self.detector is None:
            return "", []
        
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Run detection
            detection_result = self.detector.detect(mp_image)
            
            # Extract text and detections
            extracted_text, detections = self._process_detection_result(detection_result, image)
            
            return extracted_text, detections
        
        except Exception as e:
            logger.warning(f"⚠️  Text detection error: {e}")
            return "", []
    
    def _process_detection_result(self, result, original_image: np.ndarray) -> Tuple[str, List[dict]]:
        """
        Process MediaPipe detection results
        
        Args:
            result: MediaPipe detection result
            original_image: Original image for coordinate mapping
        
        Returns:
            Tuple of (text, detections_list)
        """
        text_list = []
        detections = []
        
        if result and hasattr(result, 'detections'):
            h, w = original_image.shape[:2]
            
            for detection in result.detections:
                try:
                    if hasattr(detection, 'text') and detection.text:
                        text_list.append(detection.text)
                        
                        # Extract bounding box info
                        if hasattr(detection, 'bounding_box'):
                            bbox = detection.bounding_box
                            detections.append({
                                'text': detection.text,
                                'confidence': getattr(detection, 'confidence', 0.0),
                                'bbox': {
                                    'left': int(bbox.origin_x * w) if hasattr(bbox, 'origin_x') else 0,
                                    'top': int(bbox.origin_y * h) if hasattr(bbox, 'origin_y') else 0,
                                    'width': int(bbox.width * w) if hasattr(bbox, 'width') else 0,
                                    'height': int(bbox.height * h) if hasattr(bbox, 'height') else 0,
                                }
                            })
                except Exception as e:
                    logger.debug(f"Error processing detection: {e}")
                    continue
        
        extracted_text = " ".join(text_list)
        return extracted_text, detections
    
    def detect_text_with_layout(self, image: np.ndarray) -> dict:
        """
        Detect text with layout analysis (similar to MLKit Text Recognition API)
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with text and layout information
        """
        text, detections = self.detect_text(image)
        
        # Organize text by vertical position for better reading order
        if detections:
            detections_sorted = sorted(detections, key=lambda x: (x['bbox']['top'], x['bbox']['left']))
            organized_text = " ".join([d['text'] for d in detections_sorted])
        else:
            organized_text = text
        
        return {
            'text': organized_text,
            'raw_text': text,
            'detections': detections,
            'detection_count': len(detections),
            'confidence': np.mean([d.get('confidence', 0.5) for d in detections]) if detections else 0.0
        }


def extract_text_with_mlkit_vision(image: np.ndarray, fallback_ocr=None) -> Tuple[str, dict]:
    """
    Extract text using Google MLKit Vision (MediaPipe) with fallback to traditional OCR
    
    Args:
        image: Input image as numpy array
        fallback_ocr: Optional function for fallback OCR
    
    Returns:
        Tuple of (text, metadata_dict)
    """
    extractor = GoogleMLKitVisionExtractor()
    
    # Try MediaPipe first
    layout_result = extractor.detect_text_with_layout(image)
    
    if layout_result['text'].strip():
        metadata = {
            'method': 'mediapipe',
            'confidence': layout_result['confidence'],
            'detection_count': layout_result['detection_count'],
            'fallback_used': False
        }
        return layout_result['text'], metadata
    
    # Fallback to traditional OCR if provided
    if fallback_ocr:
        text = fallback_ocr(image)
        if text.strip():
            metadata = {
                'method': 'fallback_ocr',
                'confidence': 0.0,
                'detection_count': 0,
                'fallback_used': True
            }
            return text, metadata
    
    # Return empty if no text detected
    metadata = {
        'method': 'none',
        'confidence': 0.0,
        'detection_count': 0,
        'fallback_used': False
    }
    return "", metadata


# For backward compatibility - wrapper functions
def preprocess_image_for_mlkit(image: np.ndarray, enhance=True) -> np.ndarray:
    """
    Preprocess image for MLKit Vision
    
    Args:
        image: Input image
        enhance: Whether to apply enhancement
    
    Returns:
        Preprocessed image
    """
    if enhance:
        # Gentle enhancement for better text detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # Light contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10, 10))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    return image


if __name__ == "__main__":
    # Test the extractor
    print("Testing Google MLKit Vision Extractor (MediaPipe)...")
    
    # Create a test image with text
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test News Article", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    extractor = GoogleMLKitVisionExtractor()
    text, detections = extractor.detect_text(test_image)
    
    print(f"Extracted Text: {text}")
    print(f"Detections: {detections}")
