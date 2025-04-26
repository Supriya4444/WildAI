# WILDLIFE ANIMALS IMAGE IDENTIFICATION

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import gradio as gr

# Load the model and feature extractor
model_name = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# List of animal categories in ImageNet-1k (simplified here)
animal_labels = [
    'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih-tzu', 'blenheim_spaniel', 
    'papillon', 'toy_terrier', 'rhodesian_ridgeback', 'afghan_hound', 'basset', 'beagle', 
    'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'walker_hound', 'english_foxhound', 
    'redbone', 'borzoi', 'irish_wolfhound', 'italian_greyhound', 'whippet', 'ibizan_hound', 
    'norwegian_elkhound', 'otterhound', 'saluki', 'scottish_deerhound', 'weimaraner', 'staffordshire_bullterrier',
    'american_staffordshire_terrier', 'bedlington_terrier', 'border_terrier', 'kerry_blue_terrier', 
    'irish_terrier', 'norfolk_terrier', 'norwich_terrier', 'yorkshire_terrier', 'wire-haired_fox_terrier', 
    'lakeland_terrier', 'sealyham_terrier', 'airedale', 'cairn', 'australian_terrier', 'dandie_dinmont', 
    'boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'scotch_terrier', 
    'tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier', 'west_highland_white_terrier',
    'lhasa', 'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever', 'labrador_retriever',
    'chesapeake_bay_retriever', 'german_short-haired_pointer', 'vizsla', 'english_setter', 'irish_setter',
    'gordon_setter', 'brittany_spaniel', 'clumber', 'english_springer', 'welsh_springer_spaniel',
    'cocker_spaniel', 'sussex_spaniel', 'irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 
    'malinois', 'briard', 'kelpie', 'komondor', 'old_english_sheepdog', 'shetland_sheepdog', 'collie',
    'border_collie', 'bouvier_des_flandres', 'rottweiler', 'german_shepherd', 'doberman', 'miniature_pinscher',
    'greater_swiss_mountain_dog', 'bernese_mountain_dog', 'appenzeller', 'entlebucher', 'boxer', 'bull_mastiff',
    'tibetan_mastiff', 'french_bulldog', 'great_dane', 'saint_bernard', 'eskimo_dog', 'malamute', 'siberian_husky',
    'dalmatian', 'affenpinscher', 'basenji', 'pug', 'leonberg', 'newfoundland', 'great_pyrenees', 'samoyed',
    'pomeranian', 'chow', 'keeshond', 'brabancon_griffon', 'pembroke', 'cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'mexican_hairless', 'dingo', 'red_wolf', 'grey_wolf', 'coyote', 'african_hunting_dog',
    'hyena', 'fox', 'arctic_fox', 'kit_fox', 'grey_fox', 'tabby_cat', 'tiger_cat', 'persian_cat', 'siamese_cat',
    'egyptian_cat', 'lynx', 'leopard', 'snow_leopard', 'jaguar', 'lion', 'cheetah', 'brown_bear', 'american_black_bear',
    'ice_bear', 'sloth_bear', 'mongoose', 'meerkat', 'tiger', 'elephant', 'zebra', 'giraffe', 'hippopotamus',
    'rhinoceros', 'gazelle', 'impala', 'deer', 'ibex', 'kangaroo', 'hare', 'squirrel', 'bat', 'whale', 'dolphin',
    'otter', 'seal'
]

# Function to predict species
def predict_species(image):
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx].lower()

    # Check if it matches one of our animal labels
    for animal in animal_labels:
        if animal in predicted_label:
            return f"🦁✔️ Exact Predicted Species: {animal.capitalize()}"
    return f"🦁❓ A known wild animal species (Predicted ones could be : {predicted_label})"

# Gradio interface
gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Wild Animal Species Identifier",
    description="Upload an image of a wild animal and the model will predict its species!"
).launch(share=True)
