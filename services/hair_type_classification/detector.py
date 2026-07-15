import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from utils import model_runtime as runtime

# 1. Point to your local folder where the files live
LOCAL_MODEL_PATH = "models/hair_type_classification"

# 2. Load the preprocessor and the model completely offline
# The library reads preprocessor_config.json, config.json, and model.safetensors automatically!
processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH).to(runtime.device) #type: ignore


# Put model in evaluation (inference) mode
model.eval()

def predict_hair_type(image_path):
    # 3. Load your image using PIL
    image = Image.open(image_path).convert("RGB")
    
    # 4. Preprocess the image (resizes and normalizes based on preprocessor_config.json)
    
    inputs = processor(images=image, return_tensors="pt")
    
    if runtime.is_cuda():
        inputs = {
            key: value.to(runtime.device)
            for key, value in inputs.items()
        }
    
    # 5. Run inference through the model weights (model.safetensors)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 6. Extract the highest probability index
    predicted_class_idx = logits.argmax(-1).item()
    
    # 7. Map the index back to a human-readable label using config.json's mapping
    predicted_label = model.config.id2label[predicted_class_idx] #type: ignore
    
    return predicted_label