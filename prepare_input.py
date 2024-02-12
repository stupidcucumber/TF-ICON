import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True,
                        help='Path to the folder in which results will be stored.')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the background image, from which input will be generated.')
    
    return parser.parse_args()


def binarize_image(path: str) -> cv2.Mat:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    return thresh


def generate_background(image_path: str, output_path: str):
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(512, 512))
    
    cv2.imwrite(os.path.join(output_path, 'bg01.png'), image)
    bboxes = []

    index = 0
    while True:
        x, y, w, h = cv2.selectROI('Selecting bounding box', image)
        if [x, y, w, h] == [0, 0, 0, 0]:
            cv2.destroyAllWindows()
            break

        image = cv2.rectangle(image, pt1=[x, y], pt2=[x+w, y+h], color=(0, 255, 0), thickness=2)

        mask_bg = np.zeros(shape=(512, 512))
        mask_bg[y:y+h, x:x+w] = 255

        image_path = os.path.join(output_path, 'mask_bg_fg_%d.png' % index)
        cv2.imwrite(image_path, mask_bg)

        print('Mask had been written to the file: ', image_path)
        index += 1
        bboxes.append([x, y, w, h])

    return bboxes


if __name__ == '__main__':
    args = parse_arguments()

    bboxes = generate_background(
        image_path=args.image,
        output_path=args.root
    )

    
    print('Mask extraction has been done.')
    print('Generated bboxes are: ', bboxes)