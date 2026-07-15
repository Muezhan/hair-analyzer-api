import argparse
from pathlib import Path

from services.hair_type_classification import detector
from .single_file_test import classify_single_image

testing_assets_path = Path('assets/testing/hair_type')


def main():
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        dest='mode',
    )

    # single mode
    single_parser = subparser.add_parser("single")
    single_parser.add_argument("file_path", type=str)

    # batch mode
    subparser.add_parser("batch")

    args = parser.parse_args()
    print(f"Test Arguments : {args}")
    if args.mode == 'single':
        predict_result = classify_single_image(args.file_path)
        print(f"{args.file_path} is {predict_result}")

    else:
        success_data = {}

        for hair_type in testing_assets_path.iterdir():
            hair_type_name = str(hair_type.name).lower()

            if not hair_type.is_dir():
                continue

            hair_type_data_len = len(list(hair_type.iterdir()))
            success = 0

            for hair_type_image in hair_type.iterdir():
                if not hair_type_image.is_file():
                    continue

                hair_image_name = str(hair_type_image)

                predict_result = detector.predict_hair_type(
                    hair_image_name
                )

                print(f"{hair_image_name} is {predict_result}")

                if hair_type_name == predict_result.lower():
                    success += 1

            success_data[hair_type_name] = (
                success / hair_type_data_len * 100
            )

        print("=" * 20)
        print("\nSUCCESS RATE\n")
        print("=" * 20)
        print(success_data)


if __name__ == "__main__":
    main()
else:
    main()