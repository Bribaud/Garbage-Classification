import streamlit as st
import time
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Configuration de la page
st.set_page_config(
    page_title="Classification des Déchets",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour ressembler à l'interface mobile
st.markdown("""
<style>
    .main-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 15px;
        padding: 40px 20px;
        text-align: center;
        background-color: #fafafa;
        margin: 20px 0;
    }

    .result-container {
        background-color: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
    }

    .waste-icon {
        font-size: 60px;
        margin: 20px 0;
    }

    .category-text {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin: 10px 0;
    }

    .color-text {
        font-size: 20px;
        color: #666;
        margin: 5px 0;
    }

    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #ff9500;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .button-style {
        background-color: #ff9500;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        margin: 20px 0;
    }

    .title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #333;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Définition des classes et couleurs (adaptez selon votre modèle)
CLASS_MAPPING = {
    0: {"name": "Carton", "color": "Bleu", "icon": "📦"},
    1: {"name": "Verre", "color": "Vert", "icon": "🍾"},
    2: {"name": "Métal", "color": "Jaune", "icon": "🥫"},
    3: {"name": "Papier", "color": "Bleu", "icon": "📄"},
    4: {"name": "Plastique", "color": "Jaune", "icon": "♻️"},
    5: {"name": "Ordures", "color": "Noir", "icon": "🗑️"}
}


# Classe du modèle (adaptez selon votre architecture)
class WasteClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


@st.cache_resource
def load_model():
    """Charge le modèle pré-entraîné"""
    model = WasteClassifierResNet(num_classes=6)
    # Remplacez par le chemin vers votre modèle sauvegardé
    try:
        model.load_state_dict(torch.load("/Users/ludovicbribaud/PycharmProjects/GarbageClassification/training/datasets/final_model.pth", map_location='cpu'))
        model.eval()
        return model
    except:
        st.error("Modèle non trouvé. Assurez-vous que 'model/final_model.pth' existe.")
        return None


def preprocess_image(image):
    """Préprocesse l'image pour le modèle"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)


def predict_waste(model, image):
    """Fait la prédiction sur l'image"""
    if model is None:
        return None, 0.0

    with torch.no_grad():
        preprocessed = preprocess_image(image)
        output = model(preprocessed)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    return predicted_class, confidence


def show_loading_animation():
    """Affiche l'animation de chargement"""
    loading_html = """
    <div class="result-container">
        <div class="spinner"></div>
        <p style="margin-top: 20px; color: #666;">Classification en cours...</p>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)


def show_result(predicted_class, confidence):
    """Affiche le résultat de la classification"""
    waste_info = CLASS_MAPPING[predicted_class]

    result_html = f"""
    <div class="result-container">
        <div class="waste-icon">{waste_info['icon']}</div>
        <div class="category-text">{waste_info['name']}</div>
        <div class="color-text">{waste_info['color']}</div>
        <p style="margin-top: 20px; color: #888; font-size: 14px;">
            Confiance: {confidence:.1f}%
        </p>
    </div>
    """

    return st.markdown(result_html, unsafe_allow_html=True)


def main():
    # Titre de l'application
    st.markdown('<div class="title">♻️ Classification des Déchets</div>', unsafe_allow_html=True)

    # Chargement du modèle
    model = load_model()

    # Zone d'upload
    uploaded_file = st.file_uploader(
        "Choisir une image...",
        type=['jpg', 'jpeg', 'png'],
        key="waste_image"
    )

    if uploaded_file is not None:
        # Affichage de l'image uploadée
        image = Image.open(uploaded_file)

        # Redimensionner l'image pour l'affichage
        display_image = image.copy()
        display_image.thumbnail((300, 300))

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(display_image, caption="Image à classifier", use_container_width=True)

        # Animation de chargement
        loading_placeholder = st.empty()
        result_placeholder = st.empty()

        with loading_placeholder.container():
            show_loading_animation()

        # Simulation du temps de traitement
        time.sleep(2)

        # Prédiction
        if model is not None:
            predicted_class, confidence = predict_waste(model, image)

            # Effacer l'animation de chargement
            loading_placeholder.empty()

            # Afficher le résultat
            with result_placeholder.container():
                show_result(predicted_class, confidence)

                # Bouton pour classifier une nouvelle image
                if st.button("📸 Classifier une autre image", key="new_classification"):
                    st.rerun()
        else:
            loading_placeholder.empty()
            st.error("Impossible de charger le modèle. Vérifiez le chemin du fichier.")

    else:
        # Message d'invitation
        st.markdown("""
        <div class="upload-area">
            <h3>📷 Uploadez une photo de votre déchet</h3>
            <p>Formats supportés: JPG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

        # Instructions
        st.markdown("""
        ### Comment utiliser l'application :
        1. 📱 Prenez une photo claire de votre déchet
        2. ⬆️ Uploadez l'image ci-dessus
        3. ⏳ Attendez la classification automatique
        4. ♻️ Suivez les instructions de tri affichées
        """)


if __name__ == "__main__":
    main()