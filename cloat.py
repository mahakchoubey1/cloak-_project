import cv2
import numpy as np
import time
import os
PROCESS_SCALE = 0.6         # process at smaller size (0 < scale <= 1.0). Lower -> faster, less precise.
KERNEL_SIZE = (5, 5)
BLUR_KERNEL = (5, 5)
OUTPUT_DIR = "cloak_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# default HSV ranges (these are generic starting values; use trackbars to calibrate)
default_vals = {
    "H_low": 0, "S_low": 120, "V_low": 70,
    "H_high": 10, "S_high": 255, "V_high": 255
}

# ---------- UTILS ----------
def nothing(x):
    pass

def create_trackbar_window(win_name="HSV Calibrate"):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 400, 300)
    cv2.createTrackbar("H_low", win_name, default_vals["H_low"], 180, nothing)
    cv2.createTrackbar("S_low", win_name, default_vals["S_low"], 255, nothing)
    cv2.createTrackbar("V_low", win_name, default_vals["V_low"], 255, nothing)
    cv2.createTrackbar("H_high", win_name, default_vals["H_high"], 180, nothing)
    cv2.createTrackbar("S_high", win_name, default_vals["S_high"], 255, nothing)
    cv2.createTrackbar("V_high", win_name, default_vals["V_high"], 255, nothing)
    return win_name

def read_trackbar_values(win_name="HSV Calibrate"):
    hL = cv2.getTrackbarPos("H_low", win_name)
    sL = cv2.getTrackbarPos("S_low", win_name)
    vL = cv2.getTrackbarPos("V_low", win_name)
    hH = cv2.getTrackbarPos("H_high", win_name)
    sH = cv2.getTrackbarPos("S_high", win_name)
    vH = cv2.getTrackbarPos("V_high", win_name)
    lower = np.array([hL, sL, vL])
    upper = np.array([hH, sH, vH])
    return lower, upper

def optimize_frame(frame, scale):
    if scale == 1.0:
        return frame
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return small

def upscale_frame(frame, target_shape):
    return cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cv2.namedWindow("Cloak Output", cv2.WINDOW_NORMAL)
    calibrate_win = create_trackbar_window("HSV Calibrate")
    print("Controls: 'b' = capture background, 't' = toggle trackbar window, 's' = save screenshot, 'q' = quit")

    # initial background placeholder
    background_full = None
    background_captured = False

    # morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)

    prev_time = time.time()
    fps = 0

    show_trackbars = True

    while True:
        ret, frame_full = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # flip for more natural selfie view
        frame_full = cv2.flip(frame_full, 1)

        # For faster processing, operate on scaled frame
        small = optimize_frame(frame_full, PROCESS_SCALE)
        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Read HSV bounds from trackbars
        lower_hsv, upper_hsv = read_trackbar_values("HSV Calibrate")

        # Because of scaling, we need masks at small size, then upscale the mask
        mask_small = cv2.inRange(hsv_small, lower_hsv, upper_hsv)

        # Morphological cleanup and blur on small mask
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)
        mask_small = cv2.GaussianBlur(mask_small, BLUR_KERNEL, 0)

        # Upscale mask to full resolution
        mask = upscale_frame(mask_small, frame_full.shape[:2])
        mask_inv = cv2.bitwise_not(mask)

        # Capture background when user presses 'b' (do not include cloak in background)
        # We upscale a clean background captured at full resolution
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # give small delay and capture
            print("Capturing background in 1 second... step away from frame if wearing cloak color.")
            time.sleep(1.0)
            ret_bg, bg_frame = cap.read()
            if ret_bg:
                bg_frame = cv2.flip(bg_frame, 1)
                background_full = bg_frame.copy()
                background_captured = True
                print("Background captured.")
            else:
                print("Background capture failed.")
        elif key == ord('t'):
            show_trackbars = not show_trackbars
            if show_trackbars:
                cv2.imshow("HSV Calibrate", np.zeros((10,10,3), dtype=np.uint8))  # re-show
            else:
                cv2.destroyWindow("HSV Calibrate")
        elif key == ord('s'):
            # save current result screenshot
            ts = int(time.time())
            filename = os.path.join(OUTPUT_DIR, f"cloak_demo_{ts}.jpg")
            cv2.imwrite(filename, frame_full)
            print("Saved screenshot:", filename)
        elif key == ord('q'):
            break

        if background_captured and background_full is not None:
            # Ensure background is same size
            bg = cv2.resize(background_full, (frame_full.shape[1], frame_full.shape[0]))
        else:
            # If no background captured, use a blurred version of the current frame as background fallback
            bg = cv2.GaussianBlur(frame_full, (21,21), 0)

        # Extract cloak area from background and non-cloak area from current frame
        cloaked_area = cv2.bitwise_and(bg, bg, mask=mask)         # background where cloak is detected
        non_cloak_area = cv2.bitwise_and(frame_full, frame_full, mask=mask_inv)  # original elsewhere

        # Combine final result
        result = cv2.addWeighted(non_cloak_area, 1, cloaked_area, 1, 0)

        # overlay fps
        curr_time = time.time()
        dt = curr_time - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps != 0 else (1.0 / dt)
        prev_time = curr_time
        cv2.putText(result, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # show windows
        cv2.imshow("Mask", mask)
        cv2.imshow("Cloak Output", result)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



"""import cv2  # for image processing
import numpy as np  # mathematical library for image handling
cap = cv2.VideoCapture(0)
background = cv2.imread('cloak/download.jpeg')
while cap.isOpened():
    # caturing the live frame
    ret, current_frame = cap.read()
    if ret:
        # converting from rgb to hsv color space
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        background = cv2.resize(background, (current_frame.shape[1], current_frame.shape[0]))
        # range for lower red
        lower_red1 = np.array([0, 120, 170])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        # Combine masks
        red_mask = mask1 + mask2

        # generating the final red mask
        red_mask = mask1 + mask2
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # subsituting the red portion with background image
        part1 = cv2.bitwise_and(background, background, mask=red_mask)


        # detecting things which are not red
        red_free = cv2.bitwise_not(red_mask)

        # if cloak is not present show the current image
        part2 = cv2.bitwise_and(current_frame, current_frame, mask=red_free)

        # final output
        final_output = cv2.addWeighted(part1, 1, part2, 1, 0)
        cv2.imshow("Invisibility Cloak", final_output)
        if cv2.waitKey(5) == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()"""


'''import cv2
import numpy as np
import time

# Start webcam
cap = cv2.VideoCapture(0)

#print("Adjust your camera position and make sure the green cloak is NOT visible...")
#time.sleep(2)

# Capture background (average of several frames for smoothness)
print("Capturing background...")
for i in range(60):
    ret, background = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture background frame")
background = np.flip(background, axis=1)  # Flip to mirror

print("Background captured successfully! Now wear your cloak.")

# HSV range for green cloak
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Kernel for mask cleanup
kernel = np.ones((3, 3), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)  # Mirror the webcam feed

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for green
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove noise & refine mask
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Invert mask to get the rest of the frame
    mask_inv = cv2.bitwise_not(green_mask)

    # Extract cloak area from background
    cloak_area = cv2.bitwise_and(background, background, mask=green_mask)

    # Extract rest of the scene from current frame
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    cv2.imshow("Invisibility Cloak - Green", final_output)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
