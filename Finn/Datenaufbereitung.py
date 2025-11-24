import numpy as np
from PIL import Image

uploaded_file = None  # TODO: Die zu untersuchende Datei, muss hier reingeladen werden


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")  # In RGB konvertieren
    img = img.resize((224, 224))  # Größe an Model-Anforderung anpassen
    img_array = np.array(img) / 255.0  # Pixelwerte auf 0-1 skalieren
    return img_array.reshape(1, 224, 224, 3)  # Batch-Dimension hinzufügen


img = preprocess_image(uploaded_file)

# prediction = model.predict(img)[0]  # [0], weil wir nur ein Bild haben, nicht Batch
