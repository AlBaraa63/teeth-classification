import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys

print("DEBUG: app.py is initializing...")

# Add src to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config
from models import get_transfer_model, TeethClassifierImproved

# Page Configuration
st.set_page_config(
    page_title="Teeth Classification AI",
    page_icon="🦷",
    layout="centered"
)

# Constants
MODEL_PATH = os.path.join(config.MODELS_DIR, "best_model.pth")
CLASSES = config.CLASSES
CLASS_FULL_NAMES = config.CLASS_FULL_NAMES

@st.cache_resource
def load_model():
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine which model architecture was used
    # For now, we assume it's the Transfer Learning model (ResNet18) based on recent training
    try:
        # Recreate the model architecture
        model = get_transfer_model(
            model_name='resnet18', 
            num_classes=len(CLASSES),
            pretrained=False,  # Weights will be loaded from checkpoint
            freeze_features=False
        )
        

        paths = [
            os.path.join(config.MODELS_DIR, "best_model.pth"),
            os.path.join(config.OUTPUT_DIR, "best_model.pth"),
            "best_model.pth",
            "outputs/best_model.pth",
            "outputs/models/best_model.pth"
        ]
        
        checkpoint = None
        for path in paths:
             if os.path.exists(path):
                 try:
                     print(f"DEBUG: Trying to load from {path}")
                     checkpoint = torch.load(path, map_location=device)
                     # Attempt to load state dict
                     model.load_state_dict(checkpoint['model_state_dict'])
                     
                     st.sidebar.text(f"Loaded: {os.path.basename(path)}")
                     break # Success!
                 except Exception as e:
                     print(f"DEBUG: Failed to load {path}: {e}")
                     continue
        
        if checkpoint is None:
             st.error("Could not find a valid model file. Checked: " + ", ".join(paths))
             return None, None
             
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ----------------- UI Layout -----------------

st.title("🦷 Teeth Condition Classifier")
st.markdown("Upload a dental image to classify the condition.")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This AI model classifies teeth into 7 categories:\n"
    "- Caries (Cavities)\n"
    "- Calculus (Tartar)\n"
    "- Gum Disease\n"
    "- Mouth Cancer\n"
    "- Oral Candidiasis\n"
    "- Oral Lichen Planus\n"
    "- Oral Trauma"
)
st.sidebar.markdown("---")
st.sidebar.text(f"Model: ResNet18 (Transfer Learning)")

# Main Content
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Classify Button
    if st.button("Analyze Condition"):
        with st.spinner("Analyzing..."):
            model, device = load_model()
            
            if model:
                # Prediction
                input_tensor = preprocess_image(image).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = CLASSES[predicted_idx.item()]
                full_name = CLASS_FULL_NAMES[predicted_class]
                confidence_score = confidence.item() * 100
                
                # Display Result
                st.success(f"**Prediction:** {full_name}")
                st.metric("Confidence Score", f"{confidence_score:.2f}%")
                
                # Display Top 3 probabilities
                st.markdown("### Top 3 Probabilities")
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                
                for i in range(3):
                    cls = CLASSES[top3_idx[0][i].item()]
                    prob = top3_prob[0][i].item() * 100
                    st.write(f"- **{CLASS_FULL_NAMES[cls]}**: {prob:.1f}%")
