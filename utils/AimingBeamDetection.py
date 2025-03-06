import argparse
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Argument Parsing
parser = argparse.ArgumentParser(description='Analysis of Aiming Beam detection in video files using YOLO.')
parser.add_argument('--video_path', type=str, required=True, help='Path to the video file for analysis.')
parser.add_argument('--model_name', type=str, default='best_v4n.pt', help='Name of the YOLO model file.')
args = parser.parse_args()

video_path = args.video_path
model_name = args.model_name

### example script
# python AimingBeamDetection.py --video_path "C:\Users\mm9rj\Yolov8\HN Visulization Project\AimingBeamDetection_Retro\Patient_103_Run_7.avi" --model_name "HN_v6m.pt"

def resize_mask(mask_np, orig_shape, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
    """
    Resize the mask from the transformed size back to the original size, reversing the LetterBox transformation.

    Args:
        mask_np (np.ndarray): The mask to be resized (e.g., shape (384, 640)).
        orig_shape (tuple): The original image shape (height, width).
        new_shape (tuple): The new shape used during the LetterBox transformation.
        auto (bool): Whether to use minimum rectangle (same as in LetterBox).
        scaleFill (bool): Whether to stretch the image to new_shape (same as in LetterBox).
        scaleup (bool): Whether to allow scaling up (same as in LetterBox).
        center (bool): Whether the image was centered during LetterBox transformation.
        stride (int): Stride value used in LetterBox.

    Returns:
        np.ndarray: The resized mask matching the original image size.
    """
    # Get original and new shapes
    shape = orig_shape  # Original image shape (height, width)
    mask_shape = mask_np.shape[:2]  # Current mask shape (height, width)

    # Compute scaling ratio used during LetterBox transformation
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)

    # Compute new unpadded shape
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (width, height)

    # Compute padding
    dw = new_shape[1] - new_unpad[0]  # Width padding
    dh = new_shape[0] - new_unpad[1]  # Height padding

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # Width, height ratios

    if center:
        dw /= 2
        dh /= 2

    # Compute padding amounts
    left = int(round(dw - 0.1)) if center else 0
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1)) if center else 0
    bottom = int(round(dh + 0.1))

    # Ensure padding amounts are non-negative
    left = max(left, 0)
    right = max(right, 0)
    top = max(top, 0)
    bottom = max(bottom, 0)

    # Crop the mask to remove padding
    crop_y1 = top
    crop_y2 = mask_shape[0] - bottom
    crop_x1 = left
    crop_x2 = mask_shape[1] - right

    # Handle edge cases where crop indices might be out of bounds
    crop_y2 = max(crop_y2, crop_y1)
    crop_x2 = max(crop_x2, crop_x1)

    mask_cropped = mask_np[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize the mask back to the original image size
    mask_resized = cv2.resize(mask_cropped, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask_resized

def cls_mask_to_color(cls_mask):
    custom_colors = np.array([(0, 0, 0),  # Black for background
                              (255, 0, 0),  # Red for pixel value 1
                              (0, 255, 0),  # Green for pixel value 2
                              (0, 0, 255)], dtype=np.uint8)  # Blue for pixel value 3
    color_image = custom_colors[cls_mask]
    return color_image


def arima_fill_xy(data):
    # Function to test stationarity using Augmented Dickey-Fuller test
    def test_stationarity(timeseries):
        result = adfuller(timeseries.dropna(), autolag='AIC')
        return result[1]  # p-value

    # Function to fit ARIMA model and predict missing values
    def fit_predict_arima(series):
        # Interpolate missing values for the purpose of identifying ARIMA parameters
        interpolated_series = series.interpolate(method='linear')

        # Test stationarity
        p_value = test_stationarity(interpolated_series)

        # Differencing order (d) decision based on stationarity test
        d = 0 if p_value < 0.05 else 1

        # ARIMA Model Fitting with auto AR and MA orders (p, q) using AIC
        best_aic = np.inf
        best_order = None
        best_mdl = None

        for p in range(3):  # Autoregressive lags
            for q in range(3):  # Moving average lags
                try:
                    tmp_mdl = ARIMA(interpolated_series, order=(p, d, q)).fit()
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (p, d, q)
                        best_mdl = tmp_mdl
                except:
                    continue

        print(f"Best ARIMA Order: {best_order}, AIC: {best_aic}")

        # Predicting the full series based on the best model
        predictions = best_mdl.predict(start=0, end=len(series) - 1)

        return predictions

    tmp = pd.DataFrame()
    tmp['Frame'] = data['Frame']
    tmp['X'] = fit_predict_arima(data['X'])
    tmp['Y'] = fit_predict_arima(data['Y'])

    return tmp
# Load the YOLOv8 model
model = YOLO(model_name)

video_dir = os.path.dirname(video_path)
cap = cv2.VideoCapture(video_path)
centroid_data = pd.DataFrame(columns=['Frame', 'X', 'Y'])
frame_count = 0
# Prepare VideoWriter to save the output video
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out_video = cv2.VideoWriter('merged_analysis_segmentation_mask_video.mp4', fourcc, fps, frame_size)

while cap.isOpened():

    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Access the Results object, which is the first item in the results list
        result = results[0]
        cls_mask = np.zeros((result.orig_shape[0], result.orig_shape[1]), dtype=np.uint8)
        # for result in results:
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                cls = result[i].boxes.cls
                cls = cls.cpu().numpy()
                mask_tensor = mask.data

                if mask_tensor.is_cuda:
                    mask_tensor = mask_tensor.cpu()

                mask_np = mask_tensor.numpy()

                mask_np = mask_np[0]

                mask_np = (mask_np > 0).astype(np.uint8) * 255

                mask_resized = resize_mask(mask_np, (result.orig_shape[0], result.orig_shape[1]), new_shape=mask_np.shape[:2],
                                   center=True)
                binary_mask = (mask_resized > 0).astype(np.uint8)
                # Now you can display it
                class_name = result.names[cls[0]]
                if class_name == 'AimingBeam':
                    pixel_value = 1
                elif class_name == 'Instrument':
                    pixel_value = 3
                elif class_name == 'Prob':
                    pixel_value = 2
                elif class_name == 'Fiber':
                    pixel_value = 4
                elif class_name == 'Shaft':
                    pixel_value = 5

                cls_mask[binary_mask == 1] = pixel_value

        aimingbeam = np.where(cls_mask == 1, 1, 0)
        # color_frame = cls_mask_to_color(cls_mask)

        binary_mask = np.uint8(aimingbeam * 255)

        M = cv2.moments(binary_mask)

        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = np.nan, np.nan

        centroid_data = centroid_data._append({'Frame': frame_count, 'X': centroid_x, 'Y': centroid_y},
                                              ignore_index=True)
        frame_count += 1
    #   mc_x, mc_y = motioncorrection(x,y,instrumentmask, MC=ture)
    else:
        # Break the loop if the end of the video is reached
        break
base_name = os.path.basename(video_path)
base_name = os.path.splitext(base_name)[0]
centroid_csv_path = os.path.join(video_dir, f"AimingBeam_coord_{base_name}.csv")
print("Attempting to save to:", centroid_csv_path)
centroid_data.to_csv(centroid_csv_path, index=False)
centroid_data = pd.read_csv(centroid_csv_path)
data = arima_fill_xy(centroid_data.copy())

centroid_int_csv_path = os.path.join(video_dir, f"Int_AimingBeam_coord_{base_name}.csv")
data.to_csv(centroid_int_csv_path, index=False)

cap.release()
cv2.destroyAllWindows()

# Anaysis of AimingBeam

ref_df = centroid_data
measured_df = data

# Rename columns in the reference DataFrame for clarity
ref_df.columns = ['Frame', 'X', 'Y']

total_rows = len(measured_df)

# Count rows where both X and Y contain values
non_empty_rows = ref_df.dropna(subset=['X', 'Y']).shape[0]

# Count rows where either X or Y is empty
empty_rows = total_rows - non_empty_rows

categories = ['Total Frames', 'AimingBeam Detected', 'AimingBeam Missed']
counts = [total_rows, non_empty_rows, empty_rows]

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['blue', 'green', 'red'])

plt.title('Overview of Measured Data')
plt.xlabel('Category')
plt.ylabel('Count')
plot_file_path = os.path.join(video_dir, base_name.replace('.csv', '.png'))
plt.savefig(plot_file_path)
plt.close()  # Close the plot to free up memor

# Open the video file

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(os.path.join(video_dir, f"Int_coord_augmented_{base_name}.avi"),
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

frame_count = 0
# while cap.isOpened():
for index, row in measured_df.iterrows():
    ret, frame = cap.read()
    if not ret:
        break

    if not pd.isnull(row['X']) and not pd.isnull(row['Y']):
        # Overlay a green circle at the X and Y coordinates
        cv2.circle(frame, (int(row['X']), int(row['Y'])), 15, (0, 255, 0), -1)

    out.write(frame)
    frame_count += 1

# Release everything when the job is finished

out.release()
cv2.destroyAllWindows()
print("Rendering Overlay")
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(os.path.join(video_dir, f"coord_augmented_{base_name}.avi"),
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

frame_count = 0
# while cap.isOpened():
for index, row in ref_df.iterrows():
    ret, frame = cap.read()
    if not ret:
        break

    if not pd.isnull(row['X']) and not pd.isnull(row['Y']):
        # Overlay a green circle at the X and Y coordinates
        cv2.circle(frame, (int(row['X']), int(row['Y'])), 15, (0, 255, 0), -1)

    out.write(frame)
    frame_count += 1

# Release everything when the job is finished

out.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(os.path.join(video_dir, f"coord_augmented_{base_name}.avi"),
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

frame_count = 0
# while cap.isOpened():
for index, row in ref_df.iterrows():
    ret, frame = cap.read()
    if not ret:
        break

    if not pd.isnull(row['X']) and not pd.isnull(row['Y']):
        # Overlay a green circle at the X and Y coordinates
        cv2.circle(frame, (int(row['X']), int(row['Y'])), 15, (255, 0, 0), -1)

    out.write(frame)
    frame_count += 1

# Release everything when the job is finished

out.release()
cv2.destroyAllWindows()
print("********task complete**************")
