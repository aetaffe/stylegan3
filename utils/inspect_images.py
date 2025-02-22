import cv2
from pathlib import Path
import glob
import shutil

def crop_image(img, commands):
    for command in commands:
        if command == "LB":
            img = img[:900, :, :]
        elif command == "HB":
            img = img[:820, :, :]
        elif command == "T":
            img = img[130:, :, :]
    return img

if __name__ == '__main__':
    starting_pos = 1272
    image_folder_path = '/media/alex/1TBSSD/SSD/FLImDataset/FLIm-Images-no-phantom'
    cropped_folder_path = image_folder_path.split('/')[:-1] + ['cropped']
    cropped_folder_path = '/'.join(cropped_folder_path)
    print(cropped_folder_path)
    Path(cropped_folder_path).mkdir(parents=True, exist_ok=True)

    print('\n🄸🄼🄰🄶🄴 🄲🅁🄾🄿🄿🄴🅁\n')
    print("Press '2' for low bottom crop.")
    print("Press '5' for high bottom crop.")
    print("Press '8' for top crop.")
    print("Press '0' for low bottom & top crop.")
    print("Press '7' for high bottom & top crop.")
    print("Press space bar to skip")
    print("Press 'u' to undo")
    print("Press 'b' to go back")
    print("Press 's' to save")
    print("Press 'q' to quit.")
    input("\nPress enter to start.\n\n")

    image_files = glob.glob(image_folder_path + '/*.jpg')
    image_files = sorted([int(f.split('/')[-1].split('.')[0]) for f in image_files])
    image_files = [f'{image_folder_path}/{str(f)}.jpg' for f in image_files]
    idx = starting_pos
    while idx < len(image_files):
        img_file = image_files[idx]
        image = cv2.imread(img_file)
        filename = img_file.split('/')[-1]
        # Check if the image was loaded successfully
        if image is None:
            print("Error: Could not open or find the image.")
        else:
            undo = True
            end_program = False
            while undo:
                undo = False
                skip = False
                save = False
                cv2.imshow(filename, image)
                key = cv2.waitKey(0)  # Wait for a key press
                cropped_image = image.copy()
                if key == ord('2'):
                    cropped_image = crop_image(image, ['LB'])
                elif key == ord('5'):
                    cropped_image = crop_image(image, ['HB'])
                elif key == ord('8'):
                    cropped_image = crop_image(image, ['T'])
                elif key == ord('0'):
                    cropped_image = crop_image(image, ['LB', 'T'])
                elif key == ord('7'):
                    cropped_image = crop_image(image, ['HB', 'T'])
                elif key == ord('s'):
                    save = True
                elif key == ord(' '):
                    skip = True
                    idx += 1
                elif key == ord('b'):
                    idx -= 1
                    skip = True
                elif key == ord('q'):
                    end_program = True
                    break

                if not skip and not save:
                    cv2.imshow(filename, cropped_image)
                    key = cv2.waitKey(0)

                save = key == ord('s')
                if key == ord('u'):
                    undo = True
                elif save:
                    cv2.imwrite(f'{cropped_folder_path}/{filename}', cropped_image)
                    json_file = f'{image_folder_path}/{filename.replace(".jpg", ".json")}'
                    shutil.copy(json_file, f'{cropped_folder_path}/{filename.replace(".jpg", ".json")}')
                    idx += 1

                if not undo:
                    cv2.destroyWindow(filename)
            if end_program:
                break
    cv2.destroyAllWindows()

