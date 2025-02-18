#---------------------------------------------------------------
# Loading Modules 
import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
import joblib
import pandas as pd
import pywt
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import streamlit as st
from PIL import Image
from io import BytesIO
import re
# from spellchecker import SpellChecker
# spell = SpellChecker()
from symspellpy import SymSpell, Verbosity
# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

#---------------------------------------------------------------

# Set Tesseract executable path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\e16011413\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

#--------------------------------------------------------------- 
# Load the pre-trained classification model and selected features
classification_model_all = joblib.load('classification_model_all.pkl')
selected_features_all = joblib.load('selected_features_all.pkl')

classification_model_typed = joblib.load('classification_model_typed.pkl')
selected_features_typed = joblib.load('selected_features_typed.pkl')

##print("Model classes all:", classification_model_all.classes_)
##print("Model classes typed:", classification_model_typed.classes_)
#---------------------------------------------------------------

# ---------------------------------------------------------------
# Extraction of Image and Contours 
def extract_images_from_pdf(pdf_path):
    """Extracts images from a PDF file."""
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = np.copy(img)  # Ensure the image is writable
        images.append(img)
    return images


def preprocess_image(image):
    """Preprocesses the image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding to handle varied text types
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Use a larger kernel to emphasize text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    return dilated


def merge_contours(contours, image_shape, margin=10):
    """Merges nearby contours to group fragmented text regions."""
    boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = []
    
    for box in boxes:
        x, y, w, h = box
        found_overlap = False
        for merged in merged_boxes:
            mx, my, mw, mh = merged
            if (x < mx + mw + margin and mx < x + w + margin) and \
               (y < my + mh + margin and my < y + h + margin):
                # Merge the boxes
                nx = min(mx, x)
                ny = min(my, y)
                nw = max(mx + mw, x + w) - nx
                nh = max(my + mh, y + h) - ny
                merged_boxes.remove(merged)
                merged_boxes.append((nx, ny, nw, nh))
                found_overlap = True
                break
        
        if not found_overlap:
            merged_boxes.append(box)
    
    return merged_boxes
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Extraction of Features 
def extract_text_features(roi):
    """Extracts features to distinguish between text types."""
    # 1. OCR Bounding Boxes
    boxes = pytesseract.image_to_boxes(roi)
    heights = []
    widths = []
    baselines = []
    aspect_ratios = []
    spacings = []

    prev_x2 = None  # Previous box's right boundary (x2)

    for box in boxes.splitlines():
        b = box.split()
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        width = x2 - x1
        height = y2 - y1
        baseline = y1
        aspect_ratio = width / (height if height > 0 else 1e-6)

        # Collect height, width, baseline, and aspect ratio
        heights.append(height)
        widths.append(width)
        baselines.append(baseline)
        aspect_ratios.append(aspect_ratio)

        # Calculate spacing (if there is a previous character)
        if prev_x2 is not None:
            spacings.append(x1 - prev_x2)
        prev_x2 = x2

    # 2. Calculate Variance and Consistency Features
    height_variance = np.var(heights) if heights else 0
    width_variance = np.var(widths) if widths else 0
    baseline_variance = np.var(baselines) if baselines else 0
    aspect_ratio_variance = np.var(aspect_ratios) if aspect_ratios else 0
    spacing_variance = np.var(spacings) if spacings else 0
    avg_spacing = np.mean(spacings) if spacings else 0

    # 3. Edge Features
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

    # 4. Text Features
    config = '--psm 6'  # Adjust PSM mode for better OCR
    text = pytesseract.image_to_string(roi, config=config)
    text_length = len(text.strip())

    # OCR Confidence Details
    details = pytesseract.image_to_data(roi, config=config, output_type=pytesseract.Output.DICT)
    confidence_scores = details['conf']
    avg_confidence = (
        sum([int(c) for c in confidence_scores if c != '-1']) / len(confidence_scores)
        if confidence_scores else 0
    )

    # Return features
    return {
        "height_variance": height_variance,
        "width_variance": width_variance,
        "baseline_variance": baseline_variance,
        "aspect_ratio_variance": aspect_ratio_variance,
        "spacing_variance": spacing_variance,
        "avg_spacing": avg_spacing,
        "edge_density": edge_density,
        "text_length": text_length,
        "avg_confidence": avg_confidence
    }

# Derived from the gray-level co-occurrence matrix (GLCM)
def extract_haralick_features(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return contrast, homogeneity, energy, correlation


# Entropy measures randomness or disorder in pixel intensities.
def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)

# Number of Connected Components, Size, Elongation, and Stroke Width
def connected_components_features(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    sizes = stats[:, cv2.CC_STAT_AREA]
    elongations = []
    for i in range(1, num_labels):  # Skip the background (label 0)
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elongations.append(max(w, h) / min(w, h) if min(w, h) != 0 else 0)
    return {
        "num_components": num_labels - 1,  # Exclude background
        "avg_size": np.mean(sizes[1:]) if len(sizes) > 1 else 0,
        "max_size": np.max(sizes[1:]) if len(sizes) > 1 else 0,
        "size_variance": np.var(sizes[1:]) if len(sizes) > 1 else 0,
        "avg_elongation": np.mean(elongations) if elongations else 0
    }


#Fourier Transform, Wavelet Transform
def calculate_frequency_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Wavelet Transform (using PyWavelets)
    coeffs = pywt.wavedec2(gray, 'db1', level=2)
    wavelet_features = [np.mean(coeff) for coeff in coeffs]
    return {
        "fourier_mean": np.mean(magnitude_spectrum),
        "fourier_variance": np.var(magnitude_spectrum),
        "wavelet_mean": np.mean(wavelet_features),
        "wavelet_variance": np.var(wavelet_features)
    }


# Color Features
def calculate_color_features(image):
    mean_intensity = np.mean(image)
    variance_intensity = np.var(image)
    contrast = np.max(image) - np.min(image)
    return {
        "mean_intensity": mean_intensity,
        "variance_intensity": variance_intensity,
        "contrast": contrast
    }


# Get all features into one output
def extract_features(image):
    """
    Extracts a comprehensive set of features from an image region (ROI).

    Parameters:
        image (numpy.ndarray): The image or region of interest (ROI) to extract features from.
    
    Returns:
        dict: A dictionary containing all the extracted features.
    """
    features = {}
    
    # 1. Text-Based Features
    text_features = extract_text_features(image)
    features.update(text_features)
    
    # 2. Haralick Features (Texture)
    contrast, homogeneity, energy, correlation = extract_haralick_features(image)
    features.update({
        "haralick_contrast": contrast,
        "haralick_homogeneity": homogeneity,
        "haralick_energy": energy,
        "haralick_correlation": correlation
    })
    
        
    # 3. Entropy
    entropy = calculate_entropy(image)
    features["entropy"] = entropy
    
    # 4. Connected Components Features (Requires binary image)
    binary_image = preprocess_image(image)  # Assuming preprocessing produces a binary image
    connected_features = connected_components_features(binary_image)
    features.update(connected_features)
    
    # 5. Frequency Features (Fourier and Wavelet Transform)
    frequency_features = calculate_frequency_features(image)
    features.update(frequency_features)
    
    # 6. Color Features
    color_features = calculate_color_features(image)
    features.update(color_features)
    
    return features

# ---------------------------------------------------------------
# Classification for Typed Regions
def segment_layout_typed(image):
    binary = preprocess_image(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_regions = merge_contours(contours, image.shape, margin=5)
    regions = []

    
    classification_mapping_typed = {1: 'typed-header', 0: 'typed-body'}

    # Sort merged regions from top to bottom
    merged_regions = sorted(merged_regions, key=lambda box: box[1])  # Sort by y-coordinate

    # Assign new IDs based on the sorted order
    for region_id, box in enumerate(merged_regions):
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        
        x = max(0, x - 5)
        y = max(0, y - 5)
        w = min(image.shape[1] - x, w + 10)
        h = min(image.shape[0] - y, h + 10)
        roi = image[y:y + h, x:x + w]

        features = extract_features(roi)
        feature_values = pd.DataFrame([features], columns=selected_features_typed)

        
        try:
            classification_code_typed = classification_model_typed.predict(feature_values)[0]
            classification_typed = classification_mapping_typed.get(classification_code_typed, 'unknown')
        except Exception as e:
            print(f"Error during classification: {e}")
            classification_typed = 'error'

        regions.append({
            'id': region_id,
            'x': x, 'y': y, 'width': w, 'height': h,
            'classification_typed': classification_typed,
            'features': features
        })

    return regions

def visualize_regions_typed(image, regions, output_path):
    """Visualizes the segmented regions on the image."""
    color = {
    'typed-header': (128, 0, 128),  # Purple
    'typed-body': (255, 165, 0),    # Orange
    'unknown': (255, 255, 255)      # White
    }

    for region in regions:
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        classification_typed = region['classification_typed']
        region_color = color.get(classification_typed, (255, 255, 255))  # Default to white if classification not found
        cv2.rectangle(image, (x, y), (x + w, y + h), region_color, 2)
        # cv2.putText(image, f"ID: {region['id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)

    cv2.imwrite(output_path, image)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Classification for All Regions
def segment_layout_all(image):
    binary = preprocess_image(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_regions = merge_contours(contours, image.shape, margin=5)
    regions = []

    classification_mapping_all = {0: 'handwritten', 1: 'noise', 2: 'signature', 3: 'typed', 4: 'other'}

    # Sort merged regions from top to bottom
    merged_regions = sorted(merged_regions, key=lambda box: box[1])  # Sort by y-coordinate

    # Assign new IDs based on the sorted order
    for region_id, box in enumerate(merged_regions):
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        
        x = max(0, x - 5)
        y = max(0, y - 5)
        w = min(image.shape[1] - x, w + 10)
        h = min(image.shape[0] - y, h + 10)
        roi = image[y:y + h, x:x + w]

        features = extract_features(roi)
        feature_values = pd.DataFrame([features], columns=selected_features_all)

        
        try:
            classification_code_all = classification_model_all.predict(feature_values)[0]
            classification_all = classification_mapping_all.get(classification_code_all, 'unknown')
        except Exception as e:
            print(f"Error during classification: {e}")
            classification_all = 'error'

        regions.append({
            'id': region_id,
            'x': x, 'y': y, 'width': w, 'height': h,
            'classification_all': classification_all,
            'features': features
        })

    return regions


def visualize_regions_all(image, regions, output_path):
    """Visualizes the segmented regions on the image."""

    color = {
        'handwritten': (0, 255, 0),  # Green
        'typed': (255, 0, 0),        # Blue
        'noise': (0, 0, 255),        # Red
        'signature': (0, 255, 255),  # Yellow
        'other': (128, 128, 128)     # Gray for unexpected classifications
    }

    for region in regions:
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        classification_all = region['classification_all']
        region_color = color.get(classification_all, (255, 255, 255))  # Default to white if classification not found
        cv2.rectangle(image, (x, y), (x + w, y + h), region_color, 2)
        # cv2.putText(image, f"ID: {region['id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)

    cv2.imwrite(output_path, image)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def clean_text(text):
    """Cleans extracted text by removing unnecessary line breaks, fixing OCR errors, and correcting words."""
    # Remove extra spaces and line breaks intelligently
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
    text = re.sub(r'(?<!\w)- ', '', text)  # Remove hyphenation at line breaks
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Add space between words stuck together    
    text = re.sub(r'\s*\|\s*', ' ', text)  # Remove '|'
    
    # Remove leading ‘ (or any similar character) from words
    text = re.sub(r'[‘’“”]', '', text)

    # Split text into words
    words = text.split()
    

    # Correct first word
    # if words:
        # suggestion = sym_spell.lookup(words[0], Verbosity.CLOSEST, max_edit_distance=2)
        # words[0] = suggestion[0].term if suggestion else words[0]  # Use suggestion if available


    return " ".join(words).strip()
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def extract_text_from_region(image, region):
    config = '--psm 6'  # Adjust PSM mode for better OCR
    last_section = None

    x, y, w, h = region['x'], region['y'], region['width'], region['height']
    roi = image[y:y + h, x:x + w]  # Extract region

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(roi, config=config).strip()
    extracted_text = clean_text(extracted_text)


    return extracted_text
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def reorder_header(header):
    """ Moves numerical parts of a header to the front """
    parts = header.split()  # Split words and numbers
    numbers = [part for part in parts if re.match(r'^\d', part)]  # Extract numbers
    # print ("numbers : ", numbers)
    words = [part for part in parts if not re.match(r'^\d', part)]  # Extract words
    # print ("words : ", words)
    return " ".join(numbers + words)  # Rejoin with numbers first
# ---------------------------------------------------------------

# ---------------------------------------------------------------        
def process_pdf(pdf_path):
    images = extract_images_from_pdf(pdf_path)
    typed_region_ids = set()  # Store IDs of regions classified as 'typed'


    st.write("## 🔍 Processing PDF...")

    # First Pass: Detect all Regions and Collect typed region IDs
    for page_num, image in enumerate(images):
        # print(f"Processing page {page_num + 1}...")
        st.write(f"### 📄 Processing page {page_num + 1}...")  # Streamlit display
        st.write("### 🖼️ Region Classification Color Guide")
        st.markdown("""
        - 🟩 **Green** → Handwritten Text  
        - 🟥 **Blue** → Typed Text  
        - 🟦 **Red** → Noise (irrelevant or unclassified content)  
        - 🟨 **Yellow** → Signatures  
        """)
        
        # Segment layout
        regions_all = segment_layout_all(image)

        # Identify and store typed region IDs
        for region in regions_all:
##            print(f"Region ID: {region['id']}")
##            print(f"Classification_all: {region['classification_all']}")
##            print(f"Features: {region['features']}")
            if region['classification_all'] == 'typed':
                typed_region_ids.add(region['id'])

        # Visualize regions
        output_path = f"page_{page_num + 1}_regions_all.png"
        visualize_regions_all(image, regions_all, output_path)
##        print(f"Regions visualized and saved to {output_path}")
##        print("==============================================")


    # 📷 Show processed image in Streamlit
    processed_image = cv2.imread(output_path)
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Page {page_num + 1} Processed", use_container_width=True)

    images = extract_images_from_pdf(pdf_path)
    header_body_dict = {}  # Dictionary to store paired headers and bodies


    # Second Pass: Process only regions matching stored IDs
    for page_num, image in enumerate(images):
        # print(f"Processing page for typed regions {page_num + 1}...")
        st.write(f"### 📝 Extracting typed text from page {page_num + 1}...")
        st.write("### 📑 Typed Text Segmentation Color Guide")
        st.markdown("""
        - 🔴 **Reddish-Purple** → Typed Headers (section titles or headings)  
        - 🔵 **Light Blue** → Typed Body (main text under headers)  
        """)

        # Segment layout for typed regions
        regions_typed = segment_layout_typed(image)

        # Filter regions_typed to include only IDs from typed_region_ids
        filtered_regions_typed = [region for region in regions_typed if region['id'] in typed_region_ids]


##        # Print features
##        for region in filtered_regions_typed:
##            print(f"Region ID: {region['id']}")
##            print(f"Classification_typed: {region['classification_typed']}")
##            print(f"Features: {region['features']}")
##

        # Extract text for typed regions
        for region in filtered_regions_typed:
            region['text'] = extract_text_from_region(image, region)  # Assuming function to extract text exists
##            print(f"Extracted Text (ID {region['id']}): {region['text']}")

        # Process headers and bodies
        header_text = None  # Track the current header
        current_bodies = []
        unknown_count = 1  # Track unknown headers
        last_was_header = False  # Track if the last region was a header

        for region in filtered_regions_typed:
            if region['classification_typed'] == 'typed-header':
                # If the last element was also a header, merge them
                if last_was_header:
                    # header_text += " " + region['text'].strip()# Merge with previous header
                    header_text = (header_text + " " + region['text'].strip()).strip()
                    header_text = reorder_header(header_text)  # Reorder numbers before words
                else:
                    # Store previous body before starting a new header
                    if current_bodies:
                        key = header_text if header_text else f"Unknown {unknown_count}"
                        header_body_dict[key] = " ".join(current_bodies)
                        if not header_text:
                            unknown_count += 1
                        current_bodies = []  # Reset body list
                    
                    # Start a new header
                    header_text = region['text'].strip() if region['text'].strip() else None
                    header_text = reorder_header(header_text)  # Reorder numbers before words
                    
                
                last_was_header = True  # Mark that we processed a header

            elif region['classification_typed'] == 'typed-body':
                current_bodies.append(region['text'])  # Merge consecutive bodies
                
                # **Reset header after body is encountered**
                last_was_header = False  # Body appeared, so future headers should reset

        # Store the last section if any bodies are left
        if current_bodies:
            if header_text:
                header_text = reorder_header(header_text)  # Reorder numbers before words
            key = header_text if header_text else f"Unknown {unknown_count}"
            header_body_dict[key] = " ".join(current_bodies)
            if not header_text:
                unknown_count += 1  # Increment Unknown count if needed


        # Visualize regions
        output_path = f"page_{page_num + 1}_regions_typed.png"
        visualize_regions_typed(image, filtered_regions_typed, output_path)
##        print(f"Regions_typed visualized and saved to {output_path}")
##        print("==============================================")

        # 📷 Show extracted regions in Streamlit
        processed_image = cv2.imread(output_path)
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Typed Text Regions on Page {page_num + 1}", use_container_width=True)

    # 📜 Show final extracted data
    st.write("### ✅ Extracted Headers and Bodies")
    st.write("### 📖 Extracted Data Format")
    st.markdown("""
    The extracted text is stored as a **dictionary**, where:  
    - **Keys** → Headers (typed section titles)  
    - **Values** → Corresponding body text (text under each header)
    """)
    st.json(header_body_dict)

    # print(header_body_dict)

    return header_body_dict  # Return the dictionary
# ---------------------------------------------------------------

### Example usage
# pdf_path = 'TestTitle9.pdf'  # Replace with your PDF path
# process_pdf(pdf_path)


# Streamlit App
st.title("PDF Document Region Classification & Typed Text Extraction")
st.write("Upload a PDF document to detect and classify different regions, including handwritten text, typed text, signatures, and noise. The app then extracts text from typed regions, displaying both visualized document regions and the final extracted text.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    pdf_bytes = uploaded_file.read()
    pdf_path = "uploaded_document.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Process the uploaded PDF
    process_pdf(pdf_path)

    st.success("Processing completed.") 
