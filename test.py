import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import gradio as gr

# Load model & processor
model_name = "google/vit-base-patch16-224"
extractor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Mapping ImageNet labels to clean animal species names
label_mapping = {
    'lion': 'Lion',
    'tiger_cat': 'Tiger',
    'leopard': 'Leopard',
    'cheetah': 'Cheetah',
    'jaguar': 'Jaguar',
    'brown_bear': 'Brown Bear',
    'ice_bear': 'Polar Bear',
    'zebra': 'Zebra',
    'giraffe': 'Giraffe',
    'elephant': 'Elephant',
    'grey_wolf': 'Wolf',
    'fox': 'Fox',
    'lynx': 'Lynx',
    'deer': 'Deer',
    'gazelle': 'Gazelle',
    'ibex': 'Ibex',
    'kangaroo': 'Kangaroo',
    'hare': 'Hare',
    'chimpanzee': 'Chimpanzee',
    'gorilla': 'Gorilla',
    'rhinoceros': 'Rhinoceros',
    'hippopotamus': 'Hippopotamus'
}

# Pre-filtered animal labels from ImageNet-1k that interest us
animal_labels = list(label_mapping.keys())

def predict_species(image):
    image = image.convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top5 = torch.topk(probs, 5)

    for idx in top5.indices[0]:
        label = model.config.id2label[idx.item()]
        if label in animal_labels:
            species = label_mapping[label]
            return f"✅ Wild Animal Species: {species}"

    top1_label = model.config.id2label[top5.indices[0][0].item()]
    return f"❌ Not a known wild animal species (Predicted: {top1_label.replace('_', ' ').title()})"

# Gradio interface
iface = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Wild Animal Species Identifier",
    description="Upload a wild animal image to identify its species."
)

iface.launch()