FACE_TYPE_LIST = [
    'Heart',
    'Oblong',
    'Oval',
    'Round',
    'Square'
]

class FaceTypeMaster:
    next_id = 0

    def __init__(self, name:str):
        self.name = name

        FaceTypeMaster.next_id += 1
        self.id = FaceTypeMaster.next_id


# DONT CHANGE ALL THESE CALL SEQUENCE
Heart = FaceTypeMaster('Heart')
Oblong = FaceTypeMaster('Oblong')
Oval = FaceTypeMaster('Oval')
Round = FaceTypeMaster('Round')
Square = FaceTypeMaster('Square')