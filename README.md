# About this project

This project aim to able detect the most suits hair for peoples. Its useful for some business such barbershop, salon, and more.

## Features

✓   Face shape classifier

▢   Hair type classifier

▢   Hair Visualization using AI Generated Image

## Testing.

To run all test simply run

```bash
uv run -m test
```

_*To perform all test every pre-requires need to be prepared. for more information will explained below:_

### Face Shape Classifier

#### Requirements

1.  Make folder `assets/testing/face_shape`

2.  Add folder inside created face_shape folder with name:

    -   Heart
    -   Oblong
    -   Oval
    -   Round
    -   Square
    
3.  Insert image to every face_shape childs folder.

To perform single test:

```bash
uv run -m test/face_shape_detection
```