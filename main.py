from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')


# model.predict('images/im1.png', save=True)


def detect_video(fileName):
    video_path = f"videos/{fileName}"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, save=True)
        cv2.waitKey(1)

        res_plotted = result[0].plot()
        cv2.imshow("result", res_plotted)
        out.write(res_plotted)
        if cv2.waitKey(1) == ord('q'):
            break


def detect_image(filename):
    filepath = f"images/{filename}"
    model.predict(filepath, save=True)



detect_image("im5.png")