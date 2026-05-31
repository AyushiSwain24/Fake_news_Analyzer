import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from PIL import Image, ImageEnhance
import cv2
from scipy import ndimage

# OCR libraries
try:
    import pytesseract
    HAS_PYTESS = True
except Exception:
    HAS_PYTESS = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

# Google MLKit Vision equivalent - MediaPipe
try:
    from vision_extractor import GoogleMLKitVisionExtractor, extract_text_with_mlkit_vision
    HAS_MLKIT = True
except Exception:
    HAS_MLKIT = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FactLens AI | Fake News Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (MOBILE OPTIMIZED) ---
st.markdown("""
    <style>
    * {box-sizing: border-box;}
    html, body {margin: 0; padding: 0;}
    .main {background-color: #f9f9f9; padding: 0.5rem;}
    
    /* Text input and text area optimization */
    .stTextInput input, .stTextArea textarea {
        font-size: clamp(14px, 4vw, 16px) !important;
        min-height: 40px !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea {
        min-height: 120px !important;
    }
    
    /* Button styling - full width and touch-friendly */
    .stButton>button {
        width: 100% !important;
        border-radius: 10px;
        height: clamp(44px, 12vw, 52px) !important;
        font-size: clamp(14px, 4vw, 16px) !important;
        font-weight: bold;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.75rem !important;
        margin: 0.5rem 0 !important;
    }
    .stButton>button:hover {
        background-color: #FF2B2B; 
        color: white;
    }
    
    /* Column and container adjustments */
    .stColumn {padding: 0.25rem !important;}
    .stContainer {padding: 0.5rem !important;}
    
    /* File uploader mobile fix */
    .stFileUploader {
        max-width: 100%;
    }
    .stFileUploader div {
        display: block !important;
    }
    
    /* Camera input mobile optimization */
    .stCameraInput {
        max-width: 100%;
    }
    .stCameraInput canvas {
        max-width: 100% !important;
        height: auto !important;
    }
    
    /* Tab styling for mobile */
    .stTabs [role="tablist"] {
        gap: 0.25rem;
        flex-wrap: wrap;
    }
    .stTabs [role="tab"] {
        font-size: clamp(12px, 3vw, 14px);
        padding: 0.5rem 0.75rem !important;
        margin: 0 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        height: 24px;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 0.75rem !important;
        border-radius: 8px !important;
        font-size: clamp(13px, 3.5vw, 14px) !important;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        font-size: clamp(14px, 3.5vw, 15px);
        padding: 0.5rem !important;
    }
    
    /* Responsive layout adjustments for mobile */
    @media (max-width: 768px) {
        .main {padding: 0.25rem !important;}
        .stColumn {width: 100% !important;}
        .stMetric {font-size: clamp(13px, 3.5vw, 14px) !important;}
        .stMarkdown {font-size: clamp(13px, 3.5vw, 14px) !important;}
    }
    
    /* Extra small screens */
    @media (max-width: 480px) {
        .main {padding: 0.15rem !important;}
        .stTabs [role="tab"] {
            padding: 0.4rem 0.5rem !important;
            font-size: 11px !important;
        }
        .stTextArea textarea {
            min-height: 100px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- SETUP & CACHING ---
@st.cache_resource
def download_nltk_data():
    """Download NLTK data only once and cache it"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

@st.cache_resource
def load_resources():
    """Load ML models and OCR reader with error handling"""
    # Download NLTK data
    download_nltk_data()
    
    # Import stopwords HERE to avoid "LazyLoader" errors
    from nltk.corpus import stopwords
    try:
        stop_words_set = set(stopwords.words('english'))
    except Exception as e:
        st.warning(f"Could not load stopwords: {e}")
        stop_words_set = set()

    # ML Models with better error handling - Try advanced models first, fallback to basic
    models_needed = ["model_fake.pkl", "vectorizer_fake.pkl", "model_topic.pkl", "vectorizer_topic.pkl"]
    ensemble_model = "model_fake_ensemble.pkl"
    
    # Check for models
    missing_models = [m for m in models_needed if not __import__('os').path.exists(m)]
    has_ensemble = __import__('os').path.exists(ensemble_model)
    
    if missing_models:
        return None, None, None, None, None, None, None, None
    
    try:
        model_fake = joblib.load("model_fake.pkl")
        vec_fake = joblib.load("vectorizer_fake.pkl")
        model_topic = joblib.load("model_topic.pkl")
        vec_topic = joblib.load("vectorizer_topic.pkl")
        
        # Try to load ensemble model if available
        ensemble_model_obj = None
        if has_ensemble:
            try:
                ensemble_model_obj = joblib.load(ensemble_model)
            except:
                ensemble_model_obj = None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None, None

    # Initialize OCR engines (lazy loading for performance)
    ocr_engines = {
        'tesseract': HAS_PYTESS,
        'paddle': HAS_PADDLE,
        'easyocr': HAS_EASYOCR,
        'mlkit': HAS_MLKIT,  # Google MLKit Vision (MediaPipe)
    }
    
    # Initialize MLKit vision extractor if available
    mlkit_extractor = None
    if HAS_MLKIT:
        try:
            mlkit_extractor = GoogleMLKitVisionExtractor()
        except:
            mlkit_extractor = None
    
    return model_fake, vec_fake, model_topic, vec_topic, ocr_engines, stop_words_set, ensemble_model_obj, mlkit_extractor

# Load resources
model_fake, vec_fake, model_topic, vec_topic, ocr_engines, stop_words, ensemble_model, mlkit_extractor = load_resources()

if model_fake is None:
    st.error("🚨 Models not found! Please run 'train_models.py' first to generate the .pkl files.")
    st.stop()

lemmatizer = WordNetLemmatizer()

# --- CLEANING FUNCTION (with input validation) ---
def clean_text(text, max_length=5000):
    """Clean and validate text with safety checks"""
    if not isinstance(text, str):
        return ""
    
    # Truncate extremely long inputs (prevents memory issues on mobile)
    if len(text) > max_length:
        text = text[:max_length]
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    # Use the passed stop_words set
    words = [w for w in words if w not in stop_words and len(w) > 1]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# --- MAIN APP LAYOUT ---
st.title("⚖️ FactLens AI")
st.caption("Advanced Fake News Detection & Topic Classification System")
st.divider()

# --- HELPER FUNCTIONS FOR OCR ---
def deskew_image(image):
    """Deskew image using contour analysis"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 100:
            return image
        
        angle = cv2.minAreaRect(coords)[2]
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) > 2:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_WHITE)
        
        return image
    except Exception:
        return image

def preprocess_for_ocr(image):
    """Gentle preprocessing optimized for better text extraction"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Skip deskewing if it causes issues - optional improvement
    # gray = deskew_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray)
    # if len(gray.shape) == 3:
    #     gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    # Gentle denoise with Non-Local Means (preserves edges better)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # Gentle contrast enhancement - much lower clipLimit
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10, 10))
    enhanced = clahe.apply(denoised)
    
    # Return grayscale - let OCR engines handle it as-is (better results)
    return enhanced

def preprocess_for_tesseract(image):
    """Preprocessing specifically for Tesseract (can handle binary)"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Gentle denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # Very gentle contrast
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10, 10))
    enhanced = clahe.apply(denoised)
    
    # Soft binary threshold - much gentler
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary

def extract_text_ocr(processed_image, ocr_engines, original_image=None):
    """Extract text using available OCR engines - prioritize Google MLKit Vision (MediaPipe)"""
    text_content = ""
    
    # For display: convert grayscale to RGB
    if len(processed_image.shape) == 2:
        proc_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        proc_rgb = processed_image
    
    proc_pil = Image.fromarray(cv2.cvtColor(proc_rgb, cv2.COLOR_BGR2RGB))
    
    # 0. Try Google MLKit Vision (MediaPipe) FIRST - Best overall accuracy
    if ocr_engines.get('mlkit') and mlkit_extractor is not None:
        try:
            layout_result = mlkit_extractor.detect_text_with_layout(original_image if original_image is not None else proc_rgb)
            if layout_result['text'] and layout_result['text'].strip():
                return layout_result['text'].strip(), proc_pil
        except Exception as e:
            pass
    
    # 1. Try Tesseract (best for clean document text)
    if ocr_engines.get('tesseract'):
        try:
            # Use Tesseract-specific preprocessing with binary
            if original_image is not None:
                tess_preprocessed = preprocess_for_tesseract(original_image)
                tess_pil = Image.fromarray(tess_preprocessed)
            else:
                tess_pil = proc_pil
            
            text_content = pytesseract.image_to_string(tess_pil, lang='eng')
            if text_content and text_content.strip():
                return text_content.strip(), proc_pil
        except Exception as e:
            pass
    
    # 2. Try PaddleOCR second (faster, good accuracy on grayscale)
    if ocr_engines.get('paddle'):
        try:
            paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=False)
            results = paddle_ocr.ocr(processed_image, cls=True)
            if results:
                text_content = " ".join([line[0][1] for res_line in results for line in res_line if line[0][1].strip()])
                if text_content and text_content.strip():
                    return text_content.strip(), proc_pil
        except Exception as e:
            pass
    
    # 3. Fall back to EasyOCR on original color image
    if ocr_engines.get('easyocr'):
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            if original_image is not None:
                results = reader.readtext(original_image)
            else:
                results = reader.readtext(proc_rgb)
            text_content = " ".join([res[1] for res in results if res[1].strip()])
            if text_content and text_content.strip():
                return text_content.strip(), proc_pil
        except Exception as e:
            pass
    
    return text_content, proc_pil

# --- IMAGE PROCESSING FUNCTION ---
def process_image(image_file, ocr_engines=None):
    """Process image with advanced OCR using multiple engines"""
    if image_file is None:
        return "", None

    try:
        with st.spinner("Preprocessing image for OCR..."):
            image = Image.open(image_file).convert("RGB")
            
            # Resize to reasonable size while keeping aspect ratio
            max_dim = 1280
            w, h = image.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Apply gentle preprocessing
            processed = preprocess_for_ocr(img_bgr)

        # Run OCR with multiple engines
        with st.spinner("Extracting text from image (using optimized OCR)..."):
            if ocr_engines is None:
                ocr_engines = {'tesseract': HAS_PYTESS, 'paddle': HAS_PADDLE, 'easyocr': HAS_EASYOCR}
            
            # Pass original image so each engine can use optimal preprocessing
            text_content, proc_pil = extract_text_ocr(processed, ocr_engines, original_image=img_bgr)

            if not text_content.strip():
                st.warning("No text detected. Try a clearer image or crop it to focus on text.")
                return "", proc_pil

            return text_content, proc_pil
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)[:200]}")
        return "", None

# --- IMAGE EDITING FUNCTIONS ---
def crop_image(image, crop_coords):
    """Crop image using coordinates (left, top, right, bottom)"""
    return image.crop(crop_coords)

def rotate_image(image, angle):
    """Rotate image by specified angle (degrees, counter-clockwise)"""
    return image.rotate(angle, expand=True, resample=Image.BICUBIC)

def flip_image(image, direction='horizontal'):
    """Flip image - 'horizontal' or 'vertical'"""
    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """Adjust brightness and contrast of image"""
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    return image

def resize_image(image, width=None, height=None, percent=None):
    """Resize image with various options"""
    if percent is not None:
        scale = percent / 100
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    if width and height:
        return image.resize((width, height), Image.LANCZOS)
    elif width:
        ratio = image.height / image.width
        return image.resize((width, int(width * ratio)), Image.LANCZOS)
    elif height:
        ratio = image.width / image.height
        return image.resize((int(height * ratio), height), Image.LANCZOS)
    
    return image

# --- INPUT SECTION ---
st.subheader("📖 Analyze an Article")

# Tabs for different input methods
col_tabs1, col_tabs2, col_tabs3 = st.columns(3)

with col_tabs1:
    if st.button("✍️ Text Input", use_container_width=True):
        st.session_state.current_tab = "text"
with col_tabs2:
    if st.button("🖼️ Upload Image", use_container_width=True):
        st.session_state.current_tab = "image"
with col_tabs3:
    if st.button("📸 Camera", use_container_width=True):
        st.session_state.current_tab = "camera"

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "text"

st.divider()

# Text Input Tab
if st.session_state.current_tab == "text":
    st.write("📄 Paste or type your article text:")
    def sync_text_input():
        st.session_state.article_text = st.session_state.text_input_area
    
    st.text_area(
        "Article text:",
        value=st.session_state.get("article_text", ""),
        height=150,
        placeholder="Ex: Government announces free mars vacation...",
        key="text_input_area",
        on_change=sync_text_input,
        label_visibility="collapsed"
    )

# Image Upload Tab
elif st.session_state.current_tab == "image":
    st.write("🖼️ Upload a screenshot or photo of an article:")
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    if uploaded_file:
        # Store original image
        if "uploaded_image" not in st.session_state:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
        
        edit_col1, edit_col2 = st.columns(2)
        
        with edit_col1:
            st.subheader("🎨 Edit Image")
            
            # Rotation
            rotation_angle = st.slider("Rotate (degrees)", min_value=-180, max_value=180, value=0, step=5)
            
            # Brightness & Contrast
            brightness = st.slider("Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            contrast = st.slider("Contrast", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            
            # Flip options
            col_flip1, col_flip2 = st.columns(2)
            with col_flip1:
                if st.button("↔️ Flip Horizontal", use_container_width=True):
                    st.session_state.uploaded_image = flip_image(st.session_state.uploaded_image, 'horizontal')
                    st.rerun()
            with col_flip2:
                if st.button("↕️ Flip Vertical", use_container_width=True):
                    st.session_state.uploaded_image = flip_image(st.session_state.uploaded_image, 'vertical')
                    st.rerun()
            
            # Resize
            st.write("**Resize Image:**")
            resize_col1, resize_col2 = st.columns(2)
            with resize_col1:
                resize_percent = st.number_input("Resize to % of original", min_value=10, max_value=200, value=100, step=10)
            with resize_col2:
                if st.button("Apply Resize", use_container_width=True):
                    st.session_state.uploaded_image = resize_image(st.session_state.uploaded_image, percent=resize_percent)
                    st.rerun()
            
            # Reset button
            if st.button("🔄 Reset to Original", use_container_width=True):
                if "original_uploaded_image" not in st.session_state:
                    st.session_state.original_uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.uploaded_image = st.session_state.original_uploaded_image.copy()
                st.rerun()
        
        with edit_col2:
            st.subheader("👁️ Preview")
            # Apply edits
            edited_image = st.session_state.uploaded_image.copy()
            if rotation_angle != 0:
                edited_image = rotate_image(edited_image, rotation_angle)
            if brightness != 1.0 or contrast != 1.0:
                edited_image = adjust_brightness_contrast(edited_image, brightness, contrast)
            
            st.image(edited_image, caption="Edited Preview", use_container_width=True)
        
        # Extract text from edited image
        if st.button("📄 Extract Text from Edited Image", type="primary", use_container_width=True):
            with st.spinner("Processing edited image..."):
                # Apply final edits to the image
                final_image = st.session_state.uploaded_image.copy()
                if rotation_angle != 0:
                    final_image = rotate_image(final_image, rotation_angle)
                if brightness != 1.0 or contrast != 1.0:
                    final_image = adjust_brightness_contrast(final_image, brightness, contrast)
                
                # Convert to BGR for processing
                final_image_np = np.array(final_image)
                final_image_bgr = cv2.cvtColor(final_image_np, cv2.COLOR_RGB2BGR)
                
                # Process with OCR
                extracted_text, processed_img = process_image(None, ocr_engines)
                
                # Manually run OCR on the edited image
                processed = preprocess_for_ocr(final_image_bgr)
                extracted_text, _ = extract_text_ocr(processed, ocr_engines, original_image=final_image_bgr)
                
                if processed_img is not None:
                    st.image(processed_img, caption="Processed image", use_container_width=True)
                
                if extracted_text and extracted_text.strip():
                    st.session_state.article_text = extracted_text
                    st.success("✅ Text extracted! Click to edit below.")
                else:
                    st.warning("No text detected. Try adjusting the image and try again.")
        
        if "article_text" in st.session_state and st.session_state.article_text:
            st.info(f"**Current text:** {st.session_state.article_text[:100]}...")

# Camera Tab
elif st.session_state.current_tab == "camera":
    st.write("📱 Take a photo of an article (mobile phones):")
    st.info("💡 **Mobile Tip**: Allow camera access when your browser asks. Works best on mobile phones.")
    camera_file = st.camera_input("Take a picture", label_visibility="collapsed")
    if camera_file:
        extracted_text, processed_img = process_image(camera_file, ocr_engines)
        if processed_img is not None:
            st.image(processed_img, caption="Processed image", use_container_width=True)
        if extracted_text:
            st.session_state.article_text = extracted_text
            st.success("✅ Text captured! Click to edit below.")
    if "article_text" in st.session_state and st.session_state.article_text:
        st.info(f"**Current text:** {st.session_state.article_text[:100]}...")

# Edit & Verify section (always visible)
st.divider()
st.subheader("✏️ Edit & Verify")
final_input = st.text_area(
    "Edit your text here before verification:",
    value=st.session_state.get("article_text", ""),
    height=150,
    placeholder="Your text will appear here for editing...",
    label_visibility="collapsed"
)

if st.button("🔍 Verify Authenticity", type="primary", use_container_width=True):
    text_to_analyze = final_input.strip()

    if not text_to_analyze:
        st.warning("⚠️ Please enter some text to analyze.")
    elif len(text_to_analyze) < 10:
        st.warning("⚠️ Text is too short. Please enter at least 10 characters.")
    else:
        with st.spinner("🔄 Analyzing with Advanced AI..."):
            try:
                cleaned_text = clean_text(text_to_analyze)
                
                if not cleaned_text or len(cleaned_text.split()) < 3:
                    st.warning("⚠️ Not enough valid content after cleaning. Please try different text.")
                else:
                    # Use ensemble model if available for better accuracy, otherwise use single model
                    if ensemble_model is not None:
                        try:
                            # Ensemble prediction
                            predictions, confidences = ensemble_model.predict([cleaned_text])
                            pred_label = predictions[0]
                            confidence = confidences[0] * 100
                            model_source = "Ensemble (3 ML Models)"
                        except Exception as e:
                            # Fallback to single model
                            vec_f = vec_fake.transform([cleaned_text])
                            pred_label = model_fake.predict(vec_f)[0]
                            probs = model_fake.predict_proba(vec_f)[0]
                            confidence = np.max(probs) * 100
                            model_source = "Single Model"
                    else:
                        # Use single model
                        vec_f = vec_fake.transform([cleaned_text])
                        pred_label = model_fake.predict(vec_f)[0]
                        probs = model_fake.predict_proba(vec_f)[0]
                        confidence = np.max(probs) * 100
                        model_source = "Single Model"
                    
                    # Topic classification
                    vec_t = vec_topic.transform([cleaned_text])
                    pred_topic = model_topic.predict(vec_t)[0]

                    st.markdown("### 📊 Analysis Results")
                    
                    # Create responsive columns
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if pred_label == "real":
                            st.success(f"✅ **REAL NEWS**")
                        else:
                            st.error(f"🚨 **FAKE NEWS**")
                        st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
                    
                    with col2:
                        st.info(f"📂 **Category**\n\n{pred_topic}")
                        st.progress(min(int(confidence) / 100, 1.0), text=f"{int(confidence)}%")

                    with st.expander("🔍 View Technical Details"):
                        
                        st.markdown("**Model Results:**")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prediction", pred_label.upper())
                        with col_b:
                            st.metric("Category", pred_topic)
                        with col_c:
                            st.metric("Confidence", f"{confidence:.2f}%")
                        
                        st.markdown(f"**AI Model:** {model_source}")
                        st.markdown("**Vision Technology:** Google MLKit Vision (MediaPipe) + Multi-Engine OCR")
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)[:100]}")