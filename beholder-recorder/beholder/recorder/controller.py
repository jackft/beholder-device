import configparser
import datetime
import os
import pathlib
import re
import sqlite3
import subprocess
import time

from concurrent.futures import ThreadPoolExecutor as Pool
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Union

import boto3  # type: ignore
import botocore  # type: ignore
import ffmpeg  # type: ignore
import multiprocessing_logging  # type: ignore
import psutil  # type: ignore
import requests  # type: ignore

from .computer_vision import MotionDetector
from .computer_vision import KILL_FILE as MOTION_DETECTOR_KILL_FILE
from .interval import RecordTime, IntervalCollection
from .recorder import Recorder
from .utils import pathify, _log

class S3Handler():
    def __init__(self, client: botocore.client, bucket: str):
        self.client = client
        self.bucket = bucket

    def upload(self, path: pathlib.Path, key: str) -> bool:
        if path.exists():
            _log().debug("uploading %s to bucket=%s key=%s", path, self.bucket, key)
            self.client.upload_file(str(path), self.bucket, key)
            return True
        return False

    @staticmethod
    def from_file(filename: Union[pathlib.Path, str], section="s3") -> "S3Handler":
        client = S3Handler.spaces_connect(filename, section)
        bucket = S3Handler.get_spaces_bucket(filename, section)
        return S3Handler(client, bucket)

    @staticmethod
    def spaces_connect(filename: Union[pathlib.Path, str], section="s3"):
        _keys = {"region_name", "api_version", "use_ssl", "verify",
                 "endpoint_url", "aws_access_key_id", "aws_secret_access_key",
                 "aws_session_token", "config"}
        parser = configparser.ConfigParser()
        parser.read(pathify(filename))

        session = boto3.session.Session()
        if parser.has_section(section):
            cfg = {k: v for k, v in parser.items(section) if k in _keys}
            _log().info("connected to s3")
            return session.client("s3", **cfg)
        else:
            raise Exception(f"Section {section} not found in {str(filename)}")

    @staticmethod
    def get_spaces_bucket(filename: Union[pathlib.Path, str], section="s3") -> str:
        parser = configparser.ConfigParser()
        parser.read(pathify(filename))

        if parser.has_section(section):
            return dict(parser.items(section))["bucket"]
        else:
            raise Exception(f"Section {section} not found in {str(filename)}")

class Uploader:
    av_suffixes = set([".mka", ".mkv"])
    name_pattern = re.compile(r"(\d+)_(.*)\.")

    def __init__(self,
                 s3handler: S3Handler,
                 out_path: pathlib.Path,
                 log_path: pathlib.Path,
                 prefix: pathlib.Path):
        self.s3handler = s3handler
        self.out_path = out_path
        self.log_path = log_path
        self.prefix = prefix

    @staticmethod
    def from_file(config_path):
        parser = configparser.ConfigParser(allow_no_value=True)
        with config_path.open("r") as f:
            parser.read_file(f)

        prefix = pathlib.Path(parser.get("device", "name"))
        return Uploader(
            s3handler=S3Handler.from_file(config_path),
            out_path=pathlib.Path(parser.get("recorder", "out_path")),
            log_path=pathlib.Path(parser.get("recorder", "log_path")),
            prefix=prefix
        )

    def run(self,
            process: Optional[subprocess.Popen] = None,
            now_out_path: Optional[pathlib.Path] = None):
        cnt = 0
        for path in self.out_path.glob("**/*"):
            try:
                if path.is_dir() and path != now_out_path:
                    Uploader.handle_directory(path)
                if path.suffix in Uploader.av_suffixes:
                    cnt += Uploader.handle_av_file(
                        self.prefix,
                        self.s3handler,
                        path,
                        process=process
                    )
            except Exception:
                _log().error("error handling %s", path, exc_info=True)
        for path in self.log_path.glob("**/*"):
            try:
                if path.is_dir() and path != now_out_path:
                    Uploader.handle_directory(path)
                if "log.txt" in path.name:
                    cnt += Uploader.handle_log_file(
                        self.prefix,
                        self.s3handler,
                        path
                    )
            except Exception:
                _log().error("error handling %s", path, exc_info=True)
        return cnt

    @staticmethod
    def handle_directory(path: pathlib.Path):
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

    @staticmethod
    def handle_log_file(prefix: pathlib.Path,
                        s3handler: S3Handler,
                        path: pathlib.Path):
        key = prefix / "logs" / path.name
        s3handler.upload(path, str(key))
        path.unlink()
        return 1

    @staticmethod
    def handle_av_file(prefix: pathlib.Path,
                       s3handler: S3Handler,
                       path: pathlib.Path,
                       process: Optional[subprocess.Popen] = None):
        if Uploader.is_currently_open(path, process): return 0
        try:
            creation_time = Uploader.get_creation_time(path)
            gst_start = datetime.datetime.strptime(path.parent.name, "%Y-%m-%dT%H-%M-%S")
            key_name = Uploader.keyname(gst_start, creation_time, path)
        except Exception:
            _log().error("bad av file %s", path, exc_info=True)
            path.unlink()
            return 0
        date = creation_time.strftime('%Y-%m-%d')
        hour = creation_time.strftime('%H')
        key = prefix / date / hour / key_name

        s3handler.upload(path, str(key))
        path.unlink()
        return 1

    @staticmethod
    def is_currently_open(path: pathlib.Path, process: Optional[subprocess.Popen] = None) -> bool:
        if process is not None and process.poll() is None:
            openpaths = (
                pathlib.Path(openpath.path)
                for openpath in psutil.Process(process.pid).open_files()
            )
            if any(pathlib.Path(openpath) == path for openpath in openpaths):
                return True
        return False

    @staticmethod
    def keyname(gst_start, start, file):
        match = Uploader.name_pattern.match(file.name)
        gst_group = match.groups()[0]
        source = match.groups()[1]
        gst_start_s = gst_start.strftime('%Y-%m-%dT%H-%M-%S')
        start_s = start.strftime('%Y-%m-%dT%H-%M-%S')
        return f"{start_s}.{gst_start_s}.{gst_group}.{source}{file.suffix}"

    @staticmethod
    def file2key(out_path, file) -> pathlib.Path:
        return pathlib.Path(str(file).replace(str(out_path), "").strip("/"))

    @staticmethod
    def get_creation_time(file):
        probe = ffmpeg.probe(str(file))
        creation_time = probe["format"]["tags"]["creation_time"]
        return datetime.datetime.fromisoformat(creation_time[:-1])


class Controller:
    def __init__(self,
                 recorder: Recorder,
                 uploader: Uploader,
                 motion_detector: MotionDetector,
                 database_path: str,
                 available_space_threshold: int = 500000,
                 bad_health_check_threshold: int = 10):
        self.recorder = recorder
        self.uploader = uploader
        self.motion_detector = motion_detector

        self.available_space_threshold = available_space_threshold

        self.database_conn = sqlite3.connect(
            database_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )

        self.recorder_process: Optional[subprocess.Popen] = None
        self.recorder_health_checker: Optional[RecordProcessHealthChecker] = None
        self.bad_consecutive_health_check_counter = 0
        self.bad_health_check_threshold = bad_health_check_threshold

        self.interval_collection: Optional[IntervalCollection] = self.refresh_interval_collection()
        self.record_times: Optional[RecordTime] = self.refresh_record_times()

        # stateful things
        multiprocessing_logging.install_mp_handler()
        self.upload_pool = Pool(max_workers=1)
        self.upload_future: Optional[Future] = None
        self.motion_detector_pool = Pool(max_workers=1)
        self.motion_detector_future: Optional[Future] = None

        self.uris = self.get_uris()[1]

    @staticmethod
    def from_file(config_path):
        parser = configparser.ConfigParser(allow_no_value=True)
        with config_path.open("r") as f:
            parser.read_file(f)

        return Controller(
            recorder=Recorder.from_file(config_path),
            uploader=Uploader.from_file(config_path),
            motion_detector=MotionDetector.from_file(config_path),
            database_path=parser.get("device", "database"),
            available_space_threshold=parser.getint(
                "device", "available_space_threshold", fallback=500000),
            bad_health_check_threshold=parser.get(
                "recorder", "bad_health_check_threshold", fallback=10)
        )

    def run(self, detection=True):
        while True:
            _log().debug("running loop")
            self.interval_collection = self.refresh_interval_collection()
            self.record_times = self.refresh_record_times()
            now = datetime.datetime.now()
            blackout = self.is_blackout_datetime(now)
            record_time = self.is_record_time(now)
            can_record = (
                not blackout and
                record_time
            )
            if not can_record:
                _log().debug("cannot record")
                running_reasons = []
                if not record_time:
                    running_reasons.append("not in daily record schedule")
                elif blackout:
                    running_reasons.append("disallowed by calendar event")
                self.set_running_reason(" and ".join(running_reasons))
                self.set_running_state(0)
                Controller.kill_gstreamer().wait(timeout=5)
            else:
                _log().debug("can record")
            if Controller.space_remaining() > self.available_space_threshold:
                if (
                    can_record and (
                        self.recorder_process is None or
                        self.recorder_process.poll() is not None or
                        self.uris_changed()
                    )
                ):
                    self.record()
                    self.set_running_state(1)
                    self.set_running_reason("running")
                    if detection:
                        self.start_motion_detector()
                elif can_record:
                    self.handle_recorder_health()

            if Controller.check_wifi():
                self.start_upload()

            time.sleep(60)

    def set_running_state(self, value: int):
        query = """
        INSERT INTO state(key, value) VALUES('running', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """
        cursor = self.database_conn.cursor()
        cursor.execute(query, (str(value),))
        self.database_conn.commit()

    def set_running_reason(self, value: str):
        query = """
        INSERT INTO state(key, value) VALUES('running_reason', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """
        cursor = self.database_conn.cursor()
        cursor.execute(query, (value,))
        self.database_conn.commit()

    @staticmethod
    def check_wifi() -> bool:
        try:
            requests.get("http://google.com", timeout=10)
            return True
        except (requests.ConnectionError, requests.Timeout):
            _log().error("not connected to internet", exc_info=True)
            return False

    @staticmethod
    def space_remaining() -> int:
        statvfs = os.statvfs('/')
        available_bytes = statvfs.f_frsize * statvfs.f_bavail
        return available_bytes

    def get_uris(self):
        ids: List[int] = []
        uris: List[str] = []
        names: List[str] = []
        mains: List[int] = []
        cursor = self.database_conn.cursor()
        for id, uri, name, main in cursor.execute("SELECT id, uri, name, main FROM rtspuri;"):
            ids.append(id)
            uris.append(uri)
            names.append(name)
            mains.append(main)
        return ids, uris, names, mains

    def uris_changed(self):
        _, new_uris, _, _ = self.get_uris()
        old_uris = self.uris
        _log().debug("old uris: %s", "".join(x for x in sorted(old_uris)))
        _log().debug("new uris: %s", "".join(x for x in sorted(new_uris)))
        return (
            "".join(x[1] for x in sorted(new_uris, key=lambda x: x[1]))
            !=
            "".join(x[1] for x in sorted(old_uris, key=lambda x: x[1]))
        )

    def get_main_rtsp_cam(self) -> Optional[str]:
        _, uris, _, mains = self.get_uris()
        if len(uris) == 0: return None
        rtsp = zip(uris, mains)
        maybe_uri: List[str] = [uri for uri, main in rtsp if main == 1]
        if len(maybe_uri) == 0:
            return uris[0]  # type: ignore
        return maybe_uri[0]

    def start_motion_detector(self):
        if self.motion_detector_future is not None:
            self.motion_detector_future.cancel()
            if self.motion_detector_future.running():
                MOTION_DETECTOR_KILL_FILE.touch(exist_ok=True)
                time.sleep(5)
            if self.motion_detector_future.running(): return
        uri = self.get_main_rtsp_cam()
        self.motion_detector_future = self.upload_pool.submit(lambda: self.motion_detect(uri))
        self.motion_detector_future.add_done_callback(self.motion_detector_done)

    def motion_detect(self, uri):
        _log().debug("starting motion_detector")
        if uri is None:
            _log().info("failed to start motion_detector. No main camera")
            return
        return self.motion_detector.run(uri)

    def motion_detector_done(self, future):
        if future.exception() is not None:
            _log().error("uncaught processing exception %s", future.exception(), exc_info=True)
        else:
            _log().info("motion detector stopped unexpectedly")

    def start_upload(self):
        self.upload_future = self.upload_pool.submit(self.upload)
        self.upload_future.add_done_callback(self.upload_done)

    def upload(self):
        _log().debug("starting uploader")
        return self.uploader.run(self.recorder_process, self.recorder.now_out_path)

    def upload_done(self, future):
        if future.exception() is not None:
            _log().error("uncaught processing exception %s", future.exception(), exc_info=True)
        else:
            uploaded_files = future.result()
            if uploaded_files > 0:
                _log().info("uploaded %s videos", uploaded_files)
            else:
                _log().debug("uploaded %s videos", uploaded_files)

    def record(self):
        _log().debug("stopping any recorder")
        Controller.kill_gstreamer().wait(timeout=5)
        _log().debug("starting recorder")
        time.sleep(5)
        self.recorder_process = self.recorder.run(now=True)

    @staticmethod
    def kill_gstreamer():
        return subprocess.Popen(["killall", "gst-launch-1.0"])

    @staticmethod
    def super_kill_gstreamer():
        return subprocess.Popen(["killall", "-9", "gst-launch-1.0"])

    def handle_recorder_health(self):
        if self.recorder_health_checker is None:
            self.recorder_health_checker = RecordProcessHealthChecker(self.recorder_process)
        is_healthy = self.recorder_health_checker.check()
        if is_healthy:
            self.bad_consecutive_health_check_counter = 0
            return
        _log().warn("health check: haulted %d times", self.bad_consecutive_health_check_counter)
        self.set_running_reason("possibly a bug")
        self.set_running_state(0)
        self.bad_consecutive_health_check_counter += 1
        if self.bad_consecutive_health_check_counter > self.bad_health_check_threshold:
            subprocess.Popen(["sudo reboot"])
        else:
            Controller.super_kill_gstreamer().wait(timeout=5)

    # handle record times

    def refresh_interval_collection(self):
        icol = None
        if self.database_conn is not None:
            icol = IntervalCollection.from_sql(self.database_conn.cursor())
        return icol

    def refresh_record_times(self):
        icol = None
        if self.database_conn is not None:
            icol = RecordTime.from_sql(self.database_conn.cursor())
        return icol

    def is_blackout_datetime(self, dt: datetime.datetime) -> bool:
        if self.interval_collection is not None:
            is_blackout = self.interval_collection.point_overlaps(dt)
            _log().debug("record: %s", is_blackout)
            return bool(is_blackout)
        return False

    def is_record_time(self, dt: datetime.datetime) -> bool:
        if self.record_times is not None:
            if len(self.record_times.intervals) == 0:
                return True
            is_record = self.record_times.point_overlaps(dt.time())
            _log().debug("record: %s", is_record)
            return bool(is_record)
        return True


class RecordProcessHealthChecker:
    def __init__(self, subprocess: subprocess.Popen):
        self.subprocess = subprocess
        self.process = psutil.Process(subprocess.pid)

        self.open_files: Optional[Dict[str, Any]] = None

    def check(self) -> bool:
        if self.subprocess.poll() is not None: return False
        open_files = {open_file.path: open_file for open_file in self.process.open_files()}
        if self.open_files is None:
            self.open_files = open_files
            _log().debug("health checker: new files")
            return True

        flag = False
        for path, open_file in self.open_files.items():
            # check if advancing in file
            if path in open_files:

                old_position = open_file.position
                new_position = open_files[path].position
                if old_position != new_position:
                    _log().debug("health checker: writing to file")
                    flag = True
                    break
            # probably moved on to a different file to write
            else:
                _log().debug("health checker: probably writing new files")
                flag = True
                break

        self.open_files = open_files
        return flag