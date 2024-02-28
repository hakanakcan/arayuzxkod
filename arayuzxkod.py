import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from threading import Thread
import cv2
from ultralytics import YOLO

class VideoProcessor(QtCore.QThread):
    processed = QtCore.pyqtSignal(object)

    def __init__(self, video_path, model_path, output_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Video dosyası açılamadı.")
            self.processed.emit(None)
            return

        ret, frame = cap.read()
        H, W, _ = frame.shape
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        model = YOLO(self.model_path)

        threshold = 0.5
        renk_kodlari = {
            "construction-machine": (0, 255, 255),
            "rescue-team": (255, 255, 0),
            "collapsed": (0, 0, 255),
            "solid": (0, 255, 0),
            "damaged": (0, 128, 255),
            "tilted": (0, 64, 255),
        }

        while ret:
            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    sinif_adi = results.names[int(class_id)]
                    renk = renk_kodlari[sinif_adi]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), renk, 4)
                    cv2.putText(frame, sinif_adi.upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, renk, 3, cv2.LINE_AA)

            out.write(frame)
            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.processed.emit(None)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 180, 171, 16))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(320, 180, 191, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(530, 180, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.load_video)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Lütfen Dosyayı Sürükleyiniz"))
        self.pushButton.setText(_translate("MainWindow", "Ekle"))

    def load_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(None, "Video Seç", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)

        if file_name:
            self.lineEdit.setText(file_name)
            output_path = "output_video.mp4"  # Çıkış video dosyasının adı
            model_path = "C:/Users/Hakca/OneDrive/Belgeler/Projetest/runs/detect/yolov8colab/weights/last.pt"  # Modelin yolu
            self.process_video(file_name, model_path, output_path)

    def process_video(self, video_path, model_path, output_path):
        self.video_processor = VideoProcessor(video_path, model_path, output_path)
        self.video_processor.processed.connect(self.video_processed)
        self.video_processor.start()

    def video_processed(self):
        print("Video işleme tamamlandı.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

