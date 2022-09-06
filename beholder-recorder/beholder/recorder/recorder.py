import configparser
import datetime
import pathlib
import sqlite3
import subprocess
import time

from typing import List, Optional

import ffmpeg  # type: ignore
import pulsectl  # type: ignore

from .utils import _log

def check_rtsp_health(uri: str) -> bool:
    try:
        _log().debug("running health check %s", uri)
        ffmpeg.probe(uri)
        _log().debug("health check (good)")
        return True
    except Exception:
        _log().debug("health check (bad)")
        return False
    return True


class Recorder:
    def __init__(self,
                 database_path: str,
                 out_path: str,
                 verbose: bool,
                 max_size_time=int(5 * 60 * 1e9),
                 health_check_sleep=0):
        _log().info("database %s", database_path)
        self.database_conn = sqlite3.connect(
            database_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.verbose = verbose
        self.out_path = pathlib.Path(out_path)
        self.max_size_time = max_size_time
        self.health_check_sleep = health_check_sleep
        self.now_out_path: Optional[pathlib.Path] = None

    @staticmethod
    def from_file(config_path):
        parser = configparser.ConfigParser(allow_no_value=True)
        with config_path.open("r") as f:
            parser.read_file(f)
        return Recorder(
            database_path=parser.get("device", "database"),
            verbose=parser.getboolean("recorder", "verbose", fallback=False),
            out_path=parser.get("recorder", "out_path"),
            max_size_time=parser.getint("recorder", "max_size_time", fallback=int(5 * 60 * 1e9)),
            health_check_sleep=parser.getint("recorder", "health_check_sleep", fallback=10),
        )

    @staticmethod
    def _get_pulse_source() -> Optional[str]:
        with pulsectl.Pulse('event-printer') as pulse:
            sources = sorted(
                pulse.source_list(),
                key=lambda src: src.channel_count,  # type: ignore
                reverse=True
            ) # type: ignore
        for source in sources:
            if 'usb' in source.name.lower():
                return source.index  # type: ignore
        return None

    def run(self, now=False, mp4=False):
        args = self.record_script(now=now)

        if mp4:
            args = [
                arg.replace("matroskamux", "mp4mux").replace(".mka", ".mp4").replace(".mkv", ".mp4")
                for arg in args
            ]
        print(args)
        _log().debug(" ".join(args))
        if self.health_check_sleep:
            time.sleep(self.health_check_sleep)
        p = subprocess.Popen(args)
        return p

    def record_script(self, now=False):
        script = ["gst-launch-1.0", "-e"]
        if self.verbose:
            script += ["-v"]
        script += ["clockselect", "clock-id=realtime"]
        if now:
            timedir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.now_out_path = self.out_path / timedir
            self.now_out_path.mkdir(parents=True, exist_ok=True)
        pulse_src = Recorder._get_pulse_source()
        if pulse_src is not None:
            # set volume high
            subprocess.Popen(["pacmd", "set-source-volume", str(pulse_src), "0x10000"])
            script += (
                self._record_audio_in(
                    pulse_src,
                    "mic",
                    Recorder._make_out_path(
                        self.out_path if not now else self.now_out_path,
                        "mic",
                        only_audio=True
                    )
                )
            )
        uris = set()
        for id, uri, name in zip(*self._uris()):
            _log().debug("%s %s %s", id, uri, name)
            if uri in uris: continue
            uris.add(uri)
            if (check_rtsp_health(uri)):
                script += (
                    self._record_rtsp(
                        f"{id}",
                        uri,
                        Recorder._make_out_path(
                            self.out_path if not now else self.now_out_path,
                            f"rtsp_{id}"
                        )
                    )
                )
        return script

    @staticmethod
    def _make_out_path(path, source, only_audio=False):
        return pathlib.Path(path) / f"%09d_{source}.{'mka' if only_audio else 'mkv'}"

    def _uris(self):
        ids = []
        uris = []
        names = []
        cursor = self.database_conn.cursor()
        for id, uri, name in cursor.execute("SELECT id, uri, name FROM rtspuri;"):
            ids.append(id)
            uris.append(uri)
            names.append(name)
        return ids, uris, names

    def _record_audio_in(self, device, name, out) -> List[str]:
        return (
            f'pulsesrc device="{device}"'
            f' ! audioconvert ! vorbisenc'
            f' ! mux_{name}.audio_0 splitmuxsink location="{out}" muxer=matroskamux'
            f' max-size-time={self.max_size_time} name=mux_{name} sync=false'
        ).split(" ")

    def _record_rtsp(self, name, rtsp_uri, out) -> List[str]:
        return (
            f'rtspsrc location={rtsp_uri}'
            f' is-live=true do-rtsp-keep-alive=1 do-rtcp=1 use-pipeline-clock=true name=rtsp_{name}'
            f' ! application/x-rtp, media=video, encoding-name=H264'
            f' ! rtph264depay ! h264parse ! queue'
            f' ! splitmuxsink location="{out}" muxer=matroskamux max-size-time={self.max_size_time}'
            f' name=mux_{name} sync=false '
            f'rtsp_{name}. ! rtpmp4gdepay ! aacparse ! mux_{name}.audio_0'
        ).split(" ")
