"""
Image Processor Module
Handles medical image analysis using BLIP-2 vision-language model
"""
import os
from typing import Optional, Dict
from PIL import Image
import torch

# Flag to check if transformers is available
TRANSFORMERS_AVAILABLE = False

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class ImageProcessor:
    """
    Medical Image Processor using BLIP-2
    
    Analyzes medical images and generates descriptions/insights
    """
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", use_gpu: bool = False):
        """
        Initialize the image processor
        
        Args:
            model_name: HuggingFace model name for BLIP-2
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Lazy initialization of the model (heavy operation)
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers library not available. Image processing disabled.")
            return False
        
        try:
            print(f"Loading BLIP-2 model: {self.model_name}...")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self._initialized = True
            print("BLIP-2 model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading BLIP-2 model: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if image processor is available"""
        return TRANSFORMERS_AVAILABLE
    
    def analyze_image(
        self, 
        image_path: str, 
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Analyze a medical image
        
        Args:
            image_path: Path to the image file
            prompt: Optional prompt to guide the analysis
            
        Returns:
            Analysis results
        """
        if not self.initialize():
            return {
                'success': False,
                'error': 'Image processor not available. Please install transformers library.',
                'description': None
            }
        
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Image file not found: {image_path}',
                'description': None
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Default medical analysis prompt
            if prompt is None:
                prompt = "Describe this medical image in detail, including any notable findings:"
            
            # Process image
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=200)
            
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                'success': True,
                'description': description.strip(),
                'prompt_used': prompt,
                'image_path': image_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'description': None
            }
    
    def answer_question(self, image_path: str, question: str) -> Dict:
        """
        Answer a specific question about a medical image
        
        Args:
            image_path: Path to the image file
            question: Question to answer about the image
            
        Returns:
            Answer and analysis
        """
        prompt = f"Question: {question}\nAnswer:"
        return self.analyze_image(image_path, prompt)
    
    def get_clinical_findings(self, image_path: str, image_type: str = "general") -> Dict:
        """
        Get clinical findings from a medical image
        
        Args:
            image_path: Path to the image file
            image_type: Type of medical image (xray, ct, mri, pathology, general)
            
        Returns:
            Clinical findings
        """
        prompts = {
            'xray': "Analyze this X-ray image. Describe any abnormalities, findings, and clinical significance:",
            'ct': "Analyze this CT scan image. Identify any abnormalities and describe their clinical significance:",
            'mri': "Analyze this MRI scan. Describe the visible structures and any notable findings:",
            'pathology': "Analyze this pathology/histology image. Describe the cellular structures and any abnormal findings:",
            'general': "Analyze this medical image. Provide a detailed description of what you observe and any clinical findings:"
        }
        
        prompt = prompts.get(image_type, prompts['general'])
        result = self.analyze_image(image_path, prompt)
        
        if result['success']:
            result['image_type'] = image_type
        
        return result


# Singleton instance
_image_processor = None

def get_image_processor() -> ImageProcessor:
    """Get or create image processor singleton"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor


# Simple fallback for when BLIP-2 is not available
class SimpleImageAnalyzer:
    """Fallback image analyzer that provides basic info without ML"""
    
    def analyze_image(self, image_path: str) -> Dict:
        """Basic image analysis without ML"""
        if not os.path.exists(image_path):
            return {'success': False, 'error': 'File not found'}
        
        try:
            image = Image.open(image_path)
            return {
                'success': True,
                'description': f"Image loaded successfully. Size: {image.size}, Mode: {image.mode}",
                'note': 'Full AI analysis requires BLIP-2 model. Install transformers and torch for advanced analysis.',
                'size': image.size,
                'format': image.format,
                'mode': image.mode
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
