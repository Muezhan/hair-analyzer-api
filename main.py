from models import face_shape_classification as face_shape_c
from pathlib import Path

image_assets_path = Path("assets/img")

def main():
    print("Hello from hair-analyzer-3!")
    face_shape_c.inference.main('assets/img/face_heart.jpg')

if __name__ == "__main__":
    main()