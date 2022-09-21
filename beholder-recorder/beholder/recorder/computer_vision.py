import configparser
import datetime
import pathlib
import queue
import logging
import logging.handlers
import threading
import time

from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name, cv2.CAP_FFMPEG)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def release(self):
        return self.cap.release()


KILL_FILE = pathlib.Path("/tmp/motion-detector-stop")


class MotionDetector:

    def __init__(self,
                 out_path: pathlib.Path,
                 width: int,
                 height: int,
                 learning_rate: float = 0.2,
                 percent_area: float = 0.1):
        self.out_path = out_path
        self.width = width
        self.height = height
        self.dim = (self.width, self.height)
        self.learning_rate = learning_rate
        self.percent_area = (self.width * self.height) * percent_area

        self.bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.fgmask = np.zeros((self.height, self.width), np.uint8)

        if not self.out_path.parent.exists():
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.cv_logger = logging.getLogger("cv_motion_rolling_csv_outputter")
        formatter = logging.Formatter("%(message)s")
        handler = logging.handlers.TimedRotatingFileHandler(
            self.out_path, when="m", interval=1, backupCount=5
        )
        handler.setFormatter(formatter)
        self.cv_logger.addHandler(handler)
        self.cv_logger.propagate = False
        self.cv_logger.setLevel(logging.INFO)

    @staticmethod
    def from_file(config_path):
        parser = configparser.ConfigParser(allow_no_value=True)
        with config_path.open("r") as f:
            parser.read_file(f)

        return MotionDetector(
            out_path=pathlib.Path(parser.get("recorder", "motion_detector_out_path")),
            width=parser.getint("recorder", "motion_detector_width", fallback=640),
            height=parser.getint("recorder", "motion_detector_height", fallback=480),
            learning_rate=parser.getint("recorder", "motion_detector_learning_rate", fallback=0.2),
            percent_area=parser.getint("recorder", "motion_detector_percent_area", fallback=0.01)
        )

    def motion_bboxes(self, frame) -> List[Tuple[int, int, int, int]]:
        self.fgmask = self.bgs.apply(frame, self.fgmask, self.learning_rate)
        _, absolute_difference = cv2.threshold(
            self.fgmask,
            100, 255,
            cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            absolute_difference,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        return self.get_nonzero_boxes(areas, contours)

    def get_nonzero_boxes(self, areas, contours) -> List[Tuple[int, int, int, int]]:
        if len(areas) == 0: return []
        num_boxes = min(len(areas), 3)
        indices = np.argpartition(areas, -num_boxes)[-num_boxes:]
        return [
            cv2.boundingRect(contours[idx])
            for idx in indices
            if areas[idx] > 3
        ]

    def run(self, rtsp_uri: str, test=False, sleep: Optional[float] = None):
        KILL_FILE.unlink(missing_ok=True)
        if sleep is not None:
            time.sleep(sleep)
        cap = VideoCapture(rtsp_uri)
        while True:
            if KILL_FILE.exists():
                break
            ret, frame = cap.read()
            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
            if not ret:
                break
            now = datetime.datetime.now()
            bboxes = self.motion_bboxes(frame)
            if test:
                for bbox in bboxes:
                    if bbox[2] * bbox[3] < self.percent_area: continue
                    drawbbox(frame, *bbox)
                cv2.imshow("video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            for bbox in bboxes:
                if bbox[2] * bbox[3] < self.percent_area: continue
                self.cv_logger.info(
                    "%s,%s,%s,%s,%s",
                    now.isoformat(), bbox[0], bbox[1], bbox[2], bbox[3]
                )
            time.sleep(1 / 5)
        cap.release()
        KILL_FILE.unlink(missing_ok=True)


def drawbbox(frame, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


if __name__ == "__main__":
    import sys
    md = MotionDetector(pathlib.Path("test/detect.txt"), 640, 480, 0.2, 0.01)
    md.run(sys.argv[1], test=True)
