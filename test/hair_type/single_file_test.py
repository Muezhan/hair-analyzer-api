from services.hair_type_classification import detector

def classify_single_image(path):
    return detector.predict_hair_type(path)