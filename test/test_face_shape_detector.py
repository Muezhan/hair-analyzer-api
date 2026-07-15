from services.face_shape_classfication import detector
from pathlib import Path

FACE_TYPE_LIST = [
    'Heart',
    'Oblong',
    'Oval',
    'Round',
    'Square'
]

assets_testing_path = Path('assets/testing/face_shape')
succeess_rate_data = {}

print("Starting Face Shape Testing")
for face_type in FACE_TYPE_LIST:
    face_set = assets_testing_path / face_type
    face_set_len = len(list(face_set.iterdir()))
    success = 0
    for face_sample in face_set.iterdir():
        face_shape = detector.classify_face_shape_from_path(str(face_sample.resolve()))
        print(f"{face_sample} is {face_shape.name}")
        if face_shape.name.lower() == face_type.lower():
            success += 1
    success_rate = success / face_set_len * 100
    succeess_rate_data[face_type] = success_rate
print('='*20 + "SUCCESS RATE IN PERCENTAGE\n" + '='*20)
print(succeess_rate_data)