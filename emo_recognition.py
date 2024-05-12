import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import cv2
import multiprocessing
import queue
mpl.rcParams['toolbar'] = 'None'
import tensorflow as tf
tf.get_logger().setLevel('INFO')


class EmotionDetector():
    def __init__(self) -> None:
        #self.plot = Plot(7)

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

    def predict_process(self, queue, video=0):
        self.model.load_weights('model.h5')

        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        prototxt_path = "deploy.prototxt"
        confidence_threshold = 0.5

        predictions = [0] * 7
        count = 0
        limit = 3
        maxindex = 4
        # start the webcam feed
        cap = cv2.VideoCapture(video)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             # Perform face detection
            detected_faces = self.detect_faces_dnn(frame, model_path, prototxt_path, confidence_threshold)

            # Draw rectangles around the detected faces
            for (startX, startY, endX, endY) in detected_faces:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                roi_gray = gray[startY:endY, startX:endX]
                try:
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                except cv2.error:
                    print("Face out of camera")
                    continue
                prediction = self.model.predict(cropped_img)
                predictions = [prediction[0][i]+predictions[i] for i in range(7)]
                count += 1
                if (count >= limit):
                    predictions = [round((predictions[i] / limit), 2) for i in range(7)]
                    print(predictions)

                    queue.put(predictions)

                    maxindex = int(np.argmax(predictions))
                    predictions = [0] * 7
                    count = 0

                cv2.putText(frame, emotion_dict[maxindex], (startX+20, startY-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Video', cv2.resize(frame,(720,int(720*frame.shape[0]/frame.shape[1])),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Shutting down opencv...")

    def detect_faces_dnn(self, image, model_path, prototxt_path, confidence_threshold=0.5):
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Prepare the image for detection
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        height, width = image.shape[:2]

        detected_faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                detected_faces.append((startX, startY, endX, endY))

        return detected_faces

class Plot():
    def __init__(self, theta_tick) -> None:
        self.theta_tick = theta_tick
        self.radius = 2

        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.dot, = self.ax.plot(0, 0, 'bo')
        # Format plot
        # color for polar plot and color bar
        self.yellow_tick = 0.7
        custom_cmap = self.create_custom_colormap()
        norm = mpl.colors.Normalize(0, 2*np.pi)
        n = 200  #the number of secants for the mesh
        t = np.linspace(0,2*np.pi,n)
        r = np.linspace(0,2,n)
        rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
        self.ax.pcolormesh(t, r, tg, norm=norm, cmap=custom_cmap, shading='gouraud')  #plot the colormesh on axis with colormap
        cbar = self.fig.colorbar(ScalarMappable(cmap=custom_cmap), pad=0.13)
        cbar.set_ticks([0.0, self.yellow_tick, 1.0])  # Set specific ticks for the colorbar
        cbar.set_ticklabels(['Low Certainty', 'Mid Certainty', 'High Certainty'])

        # set grid, labels
        self.ax.set_rmin(0)
        self.ax.set_rmax(self.radius)
        self.ax.set_xticks(np.pi/180. * np.linspace(0, 360, theta_tick, endpoint=False))
        self.ax.set_xticklabels(['    Neutral', 'Happy', 'Surprised', 'Fearful', 'Sad', 'Angry', ' Disgusted'])
        self.ax.set_yticklabels([])
        self.ax.set_title("Emotion Analyzer", va='bottom')
        self.ax.grid(True)

    def create_custom_colormap(self):
        # Define the colors for the colormap
        colors = [(1, 0, 0, 0.7),  # Red
                (1, 1, 0, 0.5),  # Green
                (0, 1, 1, 0.5)]  # Cyan

        # Create an array of positions for the colors along the colormap
        positions = [0.0, self.yellow_tick, 1.0]

        # Create the colormap using LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list('CustomColormap', list(zip(positions, colors)), N=256, gamma=1.0)

        return custom_cmap

    def polarize(self, data):
        length = len(data)
        polar_coords = [(2*np.pi*i/length, self.radius*data[i]) for i in range(length)]
        cartesian_coords = [(i[1] * np.cos(i[0]), i[1] * np.sin(i[0])) for i in polar_coords]
        sum_cartesian_coords = (sum(i for i, j in cartesian_coords), sum(j for i, j in cartesian_coords))
        sum_polar_coords = (round(np.arctan2(sum_cartesian_coords[1], sum_cartesian_coords[0]), 2), round(np.sqrt(sum_cartesian_coords[0]**2 + sum_cartesian_coords[1]**2), 2))
        return sum_polar_coords

    def transform(self, prediction):
        # transform the model's return prediction list to match the order we plot on the graph
        t = [0] * 7
        t[0] = prediction[4]
        t[1] = prediction[3]
        t[2] = prediction[6]
        t[3] = prediction[2]
        t[4] = prediction[5]
        t[5] = prediction[0]
        t[6] = prediction[1]
        return t

    def update_process(self, dot_queue):
        while True:
            prediction = dot_queue.get()
            prediction = self.transform(prediction)
            print(prediction, "transformed")
            theta, r = self.polarize(prediction)
            self.dot.set_xdata([theta])
            self.dot.set_ydata([r])
            print(theta,r)
            plt.pause(0.001)

if __name__ == '__main__':
    video_source = r"test4.mp4"
    ed = EmotionDetector()
    plot = Plot(7)

    queue = multiprocessing.Queue()

    # Create two processes and pass the queue as an argument
    p1 = multiprocessing.Process(target=ed.predict_process, args=(queue,), kwargs={'video': video_source})
    p2 = multiprocessing.Process(target=plot.update_process, args=(queue,))
    # Start both processes
    p1.start()
    p2.start()
