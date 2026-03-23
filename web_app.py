import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import base64

# Page config
st.set_page_config(
    page_title="FireGuard - Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with beautiful pastel background and BLACK TEXT
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #FFD1DC 0%, #B5EAD7 25%, #C7CEEA 50%, #FFB7B2 75%, #B5EAD7 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        background-attachment: fixed;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .content-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(5px);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #FF4B4B, #FF6B6B);
        color: white !important;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4);
    }
    
    .result-success {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(64, 192, 87, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .result-danger {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px dashed #2E86AB;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Slider */
    .stSlider > div {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* ALL TEXT ELEMENTS - BLACK COLOR */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    p, div, span, label {
        color: #000000 !important;
    }
    
    .stMarkdown, .stText {
        color: #000000 !important;
    }
    
    .stRadio label {
        color: #000000 !important;
        font-weight: 500;
    }
    
    .stSlider label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stFileUploader label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stFileUploader p {
        color: #000000 !important;
    }
    
    .stInfo, .stWarning, .stSuccess, .stError {
        color: #000000 !important;
    }
    
    /* Image captions */
    .stImage > div > div {
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Ensure all Streamlit text is black */
    .css-1d391kg, .css-1v0mbdj, .css-1r6slb0 {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def get_model(num_classes):
    """Initialize the Mask R-CNN model"""
    # Load pre-trained model on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    # Replace the pre-trained box predictor with a new one for our classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    
    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

@st.cache_resource
def load_model():
    """Load the trained model with enhanced error handling"""
    try:
        # Try multiple model paths
        model_paths = [
            "models/fire_detection_model.pth",
            "models/pretrained_fire_model.pth", 
            "fire_detection_model.pth",
            "best_model.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        # Initialize model with 2 classes (background + fire)
        model = get_model(2)
        
        if model_path is None:
            st.info("""
            ℹ️ **No trained model found - Using pre-trained COCO model with color detection fallback**
            
            The system will use:
            1. Pre-trained COCO model for general object detection
            2. Color-based fire detection for fire-specific detection
            """)
            # Return the pre-trained COCO model (91 classes) but we'll use it differently
            model = maskrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
        
        # Load model weights if we have a trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to allow partial loading
        else:
            # Try to load state dict directly
            try:
                model.load_state_dict(checkpoint, strict=False)
            except:
                # If that fails, try with adaptations
                model = maskrcnn_resnet50_fpn(pretrained=True)
                st.info("Using pre-trained COCO model as base")
        
        model.eval()
        if model_path:
            st.success(f"✅ Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        st.warning(f"⚠️ Using pre-trained COCO model with color detection: {str(e)}")
        # Fallback to pre-trained COCO model
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model

def detect_fire_simple(image):
    """
    Simple color-based fire detection as fallback
    This works when the model isn't trained properly
    """
    # Convert PIL to OpenCV
    img = np.array(image)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define color range for fire (red, orange, yellow)
    # Lower range for red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    # Upper range for red
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Range for orange/yellow
    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([35, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    fire_mask = mask1 + mask2 + mask3
    
    # Calculate percentage of fire-colored pixels
    total_pixels = img.shape[0] * img.shape[1]
    fire_pixels = np.count_nonzero(fire_mask)
    fire_ratio = fire_pixels / total_pixels
    
    return fire_ratio, fire_mask

def process_image_advanced(image, confidence):
    """
    Advanced processing that combines model prediction with color analysis
    """
    try:
        original_image = np.array(image)
        
        # Method 1: Try model prediction first
        model_results = None
        global model
        if model is not None:
            try:
                # Simple transform for the model
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                
                # Convert PIL to tensor and normalize
                image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    # Model expects a list of tensors
                    prediction = model([image_tensor.squeeze(0)])
                
                pred = prediction[0]
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                masks = pred['masks'].cpu().numpy()
                boxes = pred['boxes'].cpu().numpy()
                
                # Filter by confidence and look for fire-like objects
                # In COCO, fire might be detected as various classes
                # We'll consider high-confidence detections with reasonable sizes
                high_conf_idx = scores > confidence
                
                if np.any(high_conf_idx):
                    model_boxes = boxes[high_conf_idx]
                    model_masks = masks[high_conf_idx]
                    model_scores = scores[high_conf_idx]
                    
                    # Additional filtering for fire-like colors
                    fire_filtered_boxes = []
                    fire_filtered_masks = []
                    fire_filtered_scores = []
                    
                    for i, (box, mask, score) in enumerate(zip(model_boxes, model_masks, model_scores)):
                        x1, y1, x2, y2 = box.astype(int)
                        # Crop the region
                        region = original_image[y1:y2, x1:x2]
                        if region.size > 0:
                            # Check color in this region
                            fire_ratio, _ = detect_fire_simple(Image.fromarray(region))
                            if fire_ratio > 0.1:  # At least 10% fire-colored in region
                                fire_filtered_boxes.append(box)
                                fire_filtered_masks.append(mask)
                                fire_filtered_scores.append(score)
                    
                    if len(fire_filtered_boxes) > 0:
                        model_results = {
                            'boxes': np.array(fire_filtered_boxes),
                            'masks': np.array(fire_filtered_masks),
                            'scores': np.array(fire_filtered_scores),
                            'method': 'model'
                        }
                
            except Exception as e:
                # Continue to color detection if model fails
                pass
        
        # Method 2: Color-based detection (always run as backup)
        fire_ratio, fire_mask = detect_fire_simple(image)
        
        # Create bounding boxes from color detection
        color_boxes = []
        color_scores = []
        
        if fire_ratio > 0.01:  # At least 1% fire-colored pixels
            # Find contours in the fire mask
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    color_boxes.append([x, y, x + w, y + h])
                    # Use area ratio as confidence
                    confidence_score = min(area / (w * h), 0.9)
                    color_scores.append(confidence_score)
        
        # Combine or choose best results
        if model_results and len(model_results['boxes']) > 0:
            # Use model results if available
            return {
                'boxes': model_results['boxes'],
                'masks': model_results['masks'],
                'scores': model_results['scores'],
                'original_image': original_image,
                'detection_method': 'AI Model (with color validation)'
            }
        elif len(color_boxes) > 0:
            # Use color detection results
            return {
                'boxes': np.array(color_boxes),
                'masks': np.array([fire_mask] * len(color_boxes)),
                'scores': np.array(color_scores),
                'original_image': original_image,
                'detection_method': 'Color Analysis'
            }
        else:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'original_image': original_image,
                'detection_method': 'None'
            }
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def create_result_plot(original_image, results):
    """Create visualization of detection results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    fig.patch.set_facecolor('#f8f9fa')
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title("📷 Original Image", fontsize=16, fontweight='bold', pad=20, color='#2E86AB')
    ax1.axis('off')
    
    # Detection results
    ax2.imshow(original_image)
    
    if len(results['boxes']) > 0:
        for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            
            # Add label
            ax2.text(x1, y1-10, f'Fire: {score:.2f}', 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.8))
    
    ax2.set_title(f"🔥 Detection Results ({results.get('detection_method', 'Unknown')})", 
                  fontsize=16, fontweight='bold', pad=20, color='#FF4B4B')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 FireGuard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Fire Detection System</div>', unsafe_allow_html=True)
    
    # Load model
    global model
    model = load_model()
    
    # Show model status
    if model is None:
        st.warning("""
        ⚠️ **Using Color-Based Fire Detection Only**
        - The AI model is not available
        - Using color analysis (red/orange detection)
        - This works for obvious fire but may have false positives
        """)
    else:
        st.success("✅ AI Model Loaded - Using advanced fire detection with color validation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        confidence = st.slider(
            "Detection Sensitivity", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.4,  # Lower for better detection
            step=0.1,
            help="Lower = more sensitive, Higher = more strict"
        )
        
        st.markdown("---")
        st.info("""
        **How it works:**
        1. Upload any image with fire
        2. System analyzes colors and patterns
        3. Get instant fire detection results
        4. Red boxes show detected fire regions
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3>📤 Upload Image</h3>
        """, unsafe_allow_html=True)
        
        input_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Webcam", "Sample Image"],
            horizontal=True
        )
        
        image = None
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image with fire", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload images containing fire for detection"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="📁 Uploaded Image", use_column_width=True)
                
        elif input_option == "Use Webcam":
            st.info("📷 Point camera at fire source")
            picture = st.camera_input("Take a picture of fire")
            if picture is not None:
                image = Image.open(picture).convert('RGB')
                st.image(image, caption="📷 Captured Image", use_column_width=True)
                
        elif input_option == "Sample Image":
            st.info("🔬 Testing with sample fire pattern")
            # Use a sample image from the web or create a better fire-like image
            try:
                # Try to load a sample fire image if available
                sample_paths = ["sample_fire.jpg", "test_fire.jpg", "fire_sample.jpg"]
                sample_loaded = False
                for path in sample_paths:
                    if os.path.exists(path):
                        image = Image.open(path).convert('RGB')
                        sample_loaded = True
                        break
                
                if not sample_loaded:
                    # Create a synthetic fire image
                    img = np.zeros((400, 400, 3), dtype=np.uint8)
                    # Create fire-like gradient
                    for y in range(400):
                        for x in range(400):
                            # Distance from center
                            dx = x - 200
                            dy = y - 200
                            dist = np.sqrt(dx*dx + dy*dy)
                            
                            # Fire colors based on position
                            if dist < 150:
                                # Inner flame (yellow-white)
                                intensity = max(0, 255 - dist)
                                img[y, x] = [255, min(255, intensity), 0]
                            elif dist < 250:
                                # Outer flame (orange-red)
                                intensity = max(0, 200 - (dist-150))
                                img[y, x] = [255, max(50, intensity), 0]
                    
                    image = Image.fromarray(img)
                st.image(image, caption="🖼️ Sample Fire Pattern", use_column_width=True)
            except Exception as e:
                st.error(f"Could not create sample image: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3>🔍 Detection Results</h3>
        """, unsafe_allow_html=True)
        
        if image is not None:
            if st.button("🚀 Detect Fire", use_container_width=True):
                with st.spinner("🔄 Analyzing image for fire..."):
                    results = process_image_advanced(image, confidence)
                    
                    if results is not None:
                        # Display results
                        fig = create_result_plot(np.array(image), results)
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up memory
                        
                        # Show detection summary
                        if len(results['boxes']) > 0:
                            avg_score = np.mean(results['scores'])
                            st.markdown(f"""
                            <div class="result-danger">
                                <h3>🚨 FIRE DETECTED!</h3>
                                <p><strong>Fire regions found:</strong> {len(results['boxes'])}</p>
                                <p><strong>Detection method:</strong> {results.get('detection_method', 'AI Model')}</p>
                                <p><strong>Average confidence:</strong> {avg_score:.2f}</p>
                                <p><strong>Highest confidence:</strong> {np.max(results['scores']):.2f}</p>
                                <p><em>⚠️ Immediate safety measures recommended</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-success">
                                <h3>✅ NO FIRE DETECTED</h3>
                                <p><strong>Sensitivity level:</strong> {confidence}</p>
                                <p><strong>Detection method:</strong> {results.get('detection_method', 'AI Model')}</p>
                                <p>The image appears to be safe at current settings.</p>
                                <p><em>🎉 No immediate fire threat detected</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show tips for better detection
                            st.info("""
                            **💡 Tips for better detection:**
                            - Try lowering the sensitivity slider
                            - Ensure good lighting in the image
                            - Make sure fire is visible and not too small
                            - Use clear, high-quality images
                            """)
        else:
            st.info("👆 Please upload an image or use camera to start fire detection")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<strong>FireGuard</strong> | Fire Detection System | Built with PyTorch & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()