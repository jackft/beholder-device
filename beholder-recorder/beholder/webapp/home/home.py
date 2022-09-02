"""General page routes."""
import psutil  # type: ignore
import base64
import datetime
import subprocess
import pathlib
import pytz  # type: ignore

from dateutil.parser import parse
from urllib.parse import urlparse

from flask import Blueprint, render_template, jsonify, request  # type: ignore
from flask import current_app as app  # type: ignore
from flask.helpers import make_response  # type: ignore
from flask_security import login_required  # type: ignore
from flask_login import current_user  # type: ignore

from sqlalchemy import func  # type: ignore

from beholder.webapp.models import BlackoutInterval, RecordTime, RTSPURI  # type: ignore
from beholder.webapp.db import db  # type: ignore

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
    hdd = psutil.disk_usage("/")
    total = int(hdd.total / (2**30))
    used = int(hdd.used / (2**30))
    return render_template(
        "settings.jinja2",
        title="stats and settings",
        user=current_user,
        recordstart=recordstart,
        recordend=recordend,
        record_time_activated=recordactivated,
        hdd_used=used,
        hdd_total=total)

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
# Home
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
        "-vframes", "1", f"/tmp/test_{id}.jpeg"
    ])
    p.wait()
    with open(f"/tmp/test_{id}.jpeg", "rb") as f:
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

@home_bp.route('/', methods=['GET'])
@login_required
def home():
    """Home."""
    max_rtspid = db.session.query(func.max(RTSPURI.id)).scalar()
    return render_template(
        "index.jinja2",
        title="cameras",
        user=current_user,
        max_id=-1 if max_rtspid is None else max_rtspid
    )