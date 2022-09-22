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
from .recorder import check_rtsp_health, Recorder
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
                 detect_path: pathlib.Path,
                 prefix: pathlib.Path):
        self.s3handler = s3handler
        self.out_path = out_path
        self.log_path = log_path
        self.detect_path = detect_path
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
            log_path=pathlib.Path(parser.get("recorder", "log_path")).parent,
            detect_path=pathlib.Path(parser.get("recorder", "motion_detector_out_path")).parent,
            prefix=prefix
        )

    def run(self,
            process: Optional[subprocess.Popen] = None,
            now_out_path: Optional[pathlib.Path] = None):
        while True:
            vcnt = 0
            for path in self.out_path.glob("**/*"):
                try:
                    _log().debug("uploading %s", path)
                    if path.is_dir() and path != now_out_path:
                        Uploader.handle_directory(path)
                    if path.suffix in Uploader.av_suffixes:
                        vcnt += Uploader.handle_av_file(
                            self.prefix,
                            self.s3handler,
                            path,
                            process=process
                        )
                except Exception:
                    _log().error("error handling %s", path, exc_info=True)
            lcnt = 0
            for path in self.log_path.glob("**/*"):
                try:
                    _log().debug("uploading %s", path)
                    if path.name == "log.txt": continue
                    if path.is_dir() and path != now_out_path:
                        Uploader.handle_directory(path)
                    if "log.txt" in path.name:
                        lcnt += Uploader.handle_log_file(
                            self.prefix,
                            self.s3handler,
                            path
                        )
                except Exception:
                    _log().error("error handling %s", path, exc_info=True)
            dcnt = 0
            for path in self.detect_path.glob("**/*"):
                try:
                    _log().debug("uploading %s", path)
                    if path.name == "detection.txt": continue
                    if path.is_dir() and path != now_out_path:
                        Uploader.handle_directory(path)
                    if "detection.txt" in path.name:
                        dcnt += Uploader.handle_detect_file(
                            self.prefix,
                            self.s3handler,
                            path
                        )
                except Exception:
                    _log().error("error handling %s", path, exc_info=True)
            _log().debug("uploaded %d videos %s logs %d detections", vcnt, lcnt, dcnt)
            time.sleep(60)

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
    def handle_detect_file(prefix: pathlib.Path,
                           s3handler: S3Handler,
                           path: pathlib.Path):
        key = prefix / "detections" / path.name
        s3handler.upload(path, str(key))
        path.unlink()
        return 1

    @staticmethod
    def handle_av_file(prefix: pathlib.Path,
                       s3handler: S3Handler,
                       path: pathlib.Path,
                       process: Optional[subprocess.Popen] = None):
        if Uploader.is_currently_open(path): return 0
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
    def is_currently_open(path: pathlib.Path) -> bool:
        return time.time() - os.path.getmtime(str(path)) < 60

    @staticmethod
    def keyname(gst_start, start, file):
        match = Uploader.name_pattern.match(file.name)
        gst_group = match.groups()[0]
        source = match.groups()[1]
        gst_start_s = gst_start.strftime('%Y-%m-%dT%H-%M-%S')
        start_s = start.strftime('%Y-%m-%dT%H-%M-%S')
        return f"{gst_start_s}.{start_s}.{gst_group}.{source}{file.suffix}"

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
                 motion_detector_on: bool,
                 database_path: str,
                 available_space_threshold: int = 500000,
                 bad_health_check_threshold: int = 10):
        self.recorder = recorder
        self.uploader = uploader
        self.motion_detector = motion_detector
        self.motion_detector_on = motion_detector_on

        self.available_space_threshold = available_space_threshold

        self.database_conn = sqlite3.connect(
            database_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )

        self.recorder_process: Optional[subprocess.Popen] = None
        self.recorder_health_checker: Optional[RecordProcessHealthChecker] = None
        self.device_checker: Optional[DeviceChecker] = None
        self.last_health_check = None
        self.bad_consecutive_health_check_counter = 0
        self.bad_health_check_threshold = bad_health_check_threshold

        self.last_uri_check = None
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
            motion_detector_on=parser.getboolean("recorder", "motion_detector_on", fallback=False),
            database_path=parser.get("device", "database"),
            available_space_threshold=parser.getint(
                "device", "available_space_threshold", fallback=500000),
            bad_health_check_threshold=parser.getint(
                "recorder", "bad_health_check_threshold", fallback=2)
        )

    def run(self):
        while True:
            _log().debug("running loop")

            start_upload = (
                Controller.check_wifi() and
                (self.upload_future is None or self.upload_future.done())
            )
            if start_upload:
                _log().debug("starting uploader")
                self.start_upload()

            self.interval_collection = self.refresh_interval_collection()
            self.record_times = self.refresh_record_times()
            now = datetime.datetime.now()
            blackout = self.is_blackout_datetime(now)
            record_time = self.is_record_time(now)
            paused = self.get_paused_status()
            can_record = (
                not blackout and
                record_time and
                not paused
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
                if not paused:
                    Controller.kill_gstreamer().wait(timeout=5)
            else:
                _log().debug("can record")
            if Controller.space_remaining() > self.available_space_threshold:
                if (
                    can_record and (
                        self.recorder_process is None or
                        self.recorder_process.poll() is not None or
                        self.uris_changed() and
                        not self.get_paused_status()
                    )
                ):
                    self.record()
                    self.set_running_state(1)
                    self.set_running_reason("running")
                    if self.motion_detector_on:
                        self.start_motion_detector()
                elif self.get_paused_status() and self.recorder_process is not None:
                    Controller.kill_motion_detector()
                    Controller.kill_gstreamer().wait(5)
                elif can_record:
                    self.handle_recorder_health()

            time.sleep(5)

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

    def get_paused_status(self) -> bool:
        query = "SELECT value FROM state WHERE key = 'paused'"
        cursor = self.database_conn.cursor()
        cursor.execute(query)
        value = cursor.fetchone()
        if value is None:
            return False
        try:
            return bool(int(value[0]))
        except Exception:
            return False

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
        now = time.time()
        if self.last_uri_check is not None and now - self.last_uri_check < 60: return False
        _, new_uris, _, _ = self.get_uris()
        old_uris = self.uris
        _log().debug("old uris: %s", "".join(x for x in sorted(old_uris)))
        _log().debug("new uris: %s", "".join(x for x in sorted(new_uris)))
        self.last_uri_check = now
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
        self.motion_detector_future = self.upload_pool.submit(lambda: self.motion_detect(uri, 15.))
        self.motion_detector_future.add_done_callback(self.motion_detector_done)

    def motion_detect(self, uri, sleep: Optional[float] = None):
        _log().debug("starting motion_detector")
        if uri is None:
            _log().info("failed to start motion_detector. No main camera")
            return
        return self.motion_detector.run(uri, sleep=sleep)

    def motion_detector_done(self, future):
        if future.exception() is not None:
            _log().error("uncaught processing exception %s", future.exception(), exc_info=True)
        else:
            _log().info("mhandle_recorder_healthotion detector stopped unexpectedly")

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
    def kill_motion_detector():
        MOTION_DETECTOR_KILL_FILE.touch(exist_ok=True)

    @staticmethod
    def kill_gstreamer():
        return subprocess.Popen(["killall", "gst-launch-1.0"])

    @staticmethod
    def super_kill_gstreamer():
        return subprocess.Popen(["killall", "-9", "gst-launch-1.0"])

    def handle_recorder_health(self):
        if self.recorder_health_checker is None:
            self.recorder_health_checker = RecordProcessHealthChecker(self.recorder_process)
        if self.device_checker is None:
            self.device_checker = DeviceChecker(self.database_conn)
        now = time.time()
        if self.last_health_check is not None and now - self.last_health_check < 60: return
        self.last_health_check = now
        is_healthy = self.recorder_health_checker.check() and self.device_checker.check()
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
            Controller.kill_motion_detector()

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


class DeviceChecker:
    def __init__(self, database_conn):
        self.database_conn = database_conn
        self.devices: Optional[Dict[str, bool]] = None

    def check(self) -> bool:
        if self.devices is None:
            self.devices = self.refresh_devices()
            return True

        devices = self.refresh_devices()
        health = True
        for device, _health in devices.items():
            if device not in self.devices or not _health:
                health = False
        for device, _ in self.devices.items():
            if device not in devices:
                health = False
        self.devices = devices
        if not health:
            _log().warn("devices changed")
        return health

    def refresh_devices(self):
        devices = {}
        pulse_src = Recorder._get_pulse_source()
        if pulse_src is not None:
            devices[pulse_src] = True
        cursor = self.database_conn.cursor()
        for _, uri, _ in cursor.execute("SELECT id, uri, name FROM rtspuri;"):
            health = check_rtsp_health(uri)
            devices[uri] = health
        return devices


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
