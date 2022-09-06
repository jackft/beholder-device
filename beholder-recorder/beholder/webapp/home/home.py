"""General page routes."""
import psutil  # type: ignore
import base64
import datetime
import subprocess
import pathlib
import pytz  # type: ignore
import time

from dateutil.parser import parse  # type: ignore
from urllib.parse import urlparse

from flask import Blueprint, render_template, jsonify, request  # type: ignore
from flask import current_app as app  # type: ignore
from flask.helpers import make_response  # type: ignore
from flask_security import login_required  # type: ignore
from flask_login import current_user  # type: ignore

from sqlalchemy import func  # type: ignore

from beholder.webapp.models import BlackoutInterval, RecordTime, RTSPURI, State  # type: ignore
from beholder.webapp.db import db  # type: ignore
from beholder.recorder.recorder import Recorder  # type: ignore

# Blueprint Configuration
home_bp = Blueprint(
    'home_bp', __name__,
    template_folder='templates',
    static_folder='static'
)

def get_events():
    events = [
        {
            "id": e.id,
            "start": pytz.utc.localize(e.start).isoformat(),
            "end": pytz.utc.localize(e.end).isoformat(),
            "title": e.title
        } for e in BlackoutInterval.query.all()]
    app.logger.info("events %s", events)
    return events

@home_bp.route('/events/<id>', methods=['GET', 'PUT', 'DELETE'])
def event(id):
    bi = BlackoutInterval.query.get(id)
    if request.method == 'GET':
        return jsonify({"id": bi.id, "start": bi.start, "end": bi.end, "title": bi.title})
    elif request.method == 'PUT':
        row = request.json
        app.logger.info("put %s", row)
        bi.start = parse(row["start"])
        bi.end = parse(row["end"])
        bi.title = row["title"]
        db.session.commit()
        return jsonify(sucsess=True)
    elif request.method == 'DELETE':
        db.session.delete(bi)
        db.session.commit()
        return jsonify(sucsess=True)

@home_bp.route('/events', methods=['GET', 'POST'])
def events():
    if request.method == 'POST':
        row = request.json
        app.logger.info("recieved new blackout %s", row)
        bi = BlackoutInterval(
            title=row["title"],
            start=parse(row["start"]),
            end=parse(row["end"])
        )
        db.session.add(bi)
        db.session.commit()
        return jsonify(success=True, id=bi.id)
    elif request.method == 'GET':
        events = get_events()
        return jsonify(events)

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------

def get_settings_recordtime(user_id: int):
    ut = db.session.query(RecordTime).filter(RecordTime.user_id == user_id).first()
    app.logger.info("recordtime: %s", ut)
    return ut

@home_bp.route('/settings/recordtime', methods=['GET', 'PUT'])
def settings_recordtime():
    _time = get_settings_recordtime(current_user.id)
    if request.method == 'PUT':
        row = request.json
        app.logger.info("recieved new record time %s", row)
        if _time is None:
            offset = datetime.timedelta(minutes=int(row["offset"]))
            time = RecordTime(
                start=(parse(row["start"]) + offset).time(),
                end=(parse(row["end"]) + offset).time(),
                activated=row["activated"],
                user_id=current_user.id
            )
            db.session.add(time)
            db.session.commit()
            return jsonify(success=True, id=time.id)
        else:
            offset = datetime.timedelta(minutes=int(row["offset"]))
            _time.start = (parse(row["start"]) + offset).time()
            _time.end = (parse(row["end"]) + offset).time()
            _time.activated = row["activated"]
            db.session.commit()
            return jsonify(success=True, id=_time.id)
    elif request.method == 'GET':
        if _time is not None:
            return jsonify(
                {"start": _time.start.isoformat(),
                 "end": _time.end.isoformat(),
                 "activated": _time.activated})
        else:
            return jsonify(None)

@home_bp.route('/settings', methods=['GET'])
@login_required
def settings():
    recordtime = get_settings_recordtime(current_user.id)
    if recordtime is None:
        recordtime = RecordTime(
            start=parse("2:00pm").time(), # utc
            end=parse("5:00am").time(), # utc
            activated=False,
            user_id=current_user.id
        )
        db.session.add(recordtime)
        db.session.commit()
    recordstart = recordtime.start.isoformat()
    recordend = recordtime.end.isoformat()
    recordactivated = recordtime.activated
    return render_template(
        "settings.jinja2",
        title="stats and settings",
        user=current_user,
        recordstart=recordstart,
        recordend=recordend,
        record_time_activated=recordactivated)

# ------------------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------------------

@home_bp.route('/calendar', methods=['GET'])
@login_required
def calendar():
    """Calendar."""
    return render_template(
        "calendar.jinja2",
        title="calendar",
        user=current_user,
        events=get_events())


# ------------------------------------------------------------------------------
# Cameras
# ------------------------------------------------------------------------------

@home_bp.route('/rtspuri/<int:id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def rtspuri(id):
    rtsp = RTSPURI.query.get(id)
    if request.method == 'GET':
        return jsonify({"id": rtsp.id, "name": rtsp.name, "uri": rtsp.uri})
    elif request.method == 'PUT':
        row = request.json
        app.logger.info("put %s", row)
        uris = RTSPURI.query.all()
        hosts = {urlparse(uri.uri).hostname: uri.id for uri in uris}
        hostname = urlparse(row["uri"]).hostname
        if hostname in hosts and hosts[hostname] != id:
            app.logger.info(hosts)
            return jsonify(success=False, error="duplicate host"), 400
        rtsp.uri = row["uri"]
        rtsp.name = row["name"]
        db.session.commit()
        return jsonify(sucsess=True), 200
    elif request.method == 'DELETE':
        db.session.delete(rtsp)
        db.session.commit()
        return jsonify(sucsess=True)


@home_bp.route('/test_rtsp_uri', methods=['POST'])
@login_required
def test_rtsp_uri():
    row = request.json
    uri = row["uri"]
    id = row["id"]
    p = pathlib.Path(f"/tmp/test_{id if id is not None else 'test'}.jpeg")
    if p.exists():
        p.unlink()
    p = subprocess.Popen([
        "ffmpeg", "-y",
        "-i", uri,
        "-vframes", "1", str(p)
    ])
    p.wait()
    with p.open("rb") as f:
        image_binary = f.read()

    response = make_response(base64.b64encode(image_binary))
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='image.jpeg')
    return response


@home_bp.route('/rtspuri', methods=['GET', 'POST'])
@login_required
def rtspuris():
    if request.method == 'POST':
        rtsp = request.json
        app.logger.info("recieved new rtsp %s", rtsp)
        uris = RTSPURI.query.all()
        hosts = {urlparse(uri.uri).hostname for uri in uris}
        if urlparse(rtsp["uri"]).hostname in hosts:
            app.logger.info(hosts)
            return jsonify(success=False, error="duplicate host"), 400
        row = RTSPURI(
            uri=rtsp["uri"],
            name=rtsp["name"]
        )
        db.session.add(row)
        db.session.commit()
        return jsonify(success=True, id=row.id)
    elif request.method == 'GET':
        uris = RTSPURI.query.all()
        return jsonify(
            [
                {
                    "id": uri.id,
                    "uri": uri.uri,
                    "name": uri.name
                }
                for uri in uris
            ]
        )

@home_bp.route('/cameras', methods=['GET'])
@login_required
def cameras():
    """cameras."""
    max_rtspid = db.session.query(func.max(RTSPURI.id)).scalar()
    return render_template(
        "cameras.jinja2",
        title="cameras",
        user=current_user,
        max_id=-1 if max_rtspid is None else max_rtspid
    )

# ------------------------------------------------------------------------------
# Home
# ------------------------------------------------------------------------------

@home_bp.route('/state', methods=['GET'])
@login_required
def state():
    running_state = db.session.query(State).filter(State.key == "running").first()
    running_reason = db.session.query(State).filter(State.key == "running_reason").first()

    gst_procs = len([p for p in psutil.process_iter() if 'gst-launch' in p.name().lower()])

    if running_state is None:
        running_state = "-"
    elif running_state.value == "1" and gst_procs > 0:
        running_state = "running"
    else:
        running_state = "not running"

    if running_reason is None or gst_procs == 0:
        running_reason = "-"
    else:
        print(running_reason.updated)
        running_reason = running_reason.value

    return jsonify({
        "running_state": running_state,
        "running_reason": running_reason,
    })


@home_bp.route('/test', methods=['GET'])
@login_required
def test():
    recorder = Recorder.from_file(pathlib.Path(app.config["CONFIG"]))
    recorder.out_path = "/tmp/test"
    path = pathlib.Path(recorder.out_path)
    path.mkdir(parents=True, exist_ok=True)

    for child in path.glob("*"):
        child.unlink()

    subprocess.Popen(["killall", "-9", "gst-launch-1.0"]).wait(2)
    recorder.run()
    time.sleep(15)
    subprocess.Popen(["killall", "-9", "gst-launch-1.0"]).wait(2)
    videos = []
    for child in path.glob("*.mkv"):
        video = path / child.name.replace(child.suffix, ".mp4")
        subprocess.Popen([
            "ffmpeg", "-y",
            "-i", str(child),
            "-c:v", "copy",
            str(video)
        ]).wait(timeout=5)
        videos.append(video)
    audios = []
    for child in path.glob("*.mka"):
        audio = path / child.name.replace(child.suffix, ".mp3")
        subprocess.Popen([
            "ffmpeg", "-y",
            "-i", str(child),
            str(audio)
        ]).wait(timeout=5)
        audios.append(audio)
    return jsonify(
        {
            "videos": [f"/video/{video.name}" for video in videos],
            "audios": [f"/video/{audio.name}" for audio in audios]
        }
    )

@home_bp.route('/', methods=['GET'])
@login_required
def home():
    """home."""
    hdd = psutil.disk_usage("/")
    total = int(hdd.total / (2**30))
    used = int(hdd.used / (2**30))
    return render_template(
        "index.jinja2",
        title="home",
        user=current_user,
        hdd_used=used,
        hdd_total=total
    )
