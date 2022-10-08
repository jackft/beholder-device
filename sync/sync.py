import datetime
import logging
import pathlib
import sqlite3
import sys
from typing import List, Optional, Union
from ctypes import c_int64
import subprocess

from dateutil import parser

import pandas as pd
import click
from click_loglevel import LogLevel  # type: ignore
import ffmpeg  # type: ignore
from dateutil import parser

from audio_offset_finder.audio_offset_finder import find_offset_between_files

from ffmpeg_wrapper import Concat, Trim, SetPTS, FilterComplex, Movie, ASetPTS, ATrim, XStack, DummyMovie, NullSink, ANullSink


def _log() -> logging.Logger:
    return logging.getLogger()

def setup_logging(level=logging.INFO):
    root = logging.getLogger()
    formatter = logging.Formatter(
        "[%(asctime)s %(process)d] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = True
    root.setLevel(level)


class Media:
    def __init__(self, path: Union[pathlib.Path, str]):
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists(): assert False
        self.path = path
        self.name = path.name.split(".")[-2]
        self._obj = ffmpeg.probe(str(path))

        self.creation_time = Media.get_creation_time(self._obj)
        self.duration = Media.get_duration(self._obj)
        self.video_offset = Media.get_video_offset(self._obj)
        self.audio_offset = Media.get_audio_offset(self._obj)

    def offset(self, other: "Media"):
        offset = find_offset_between_files(str(self.path), str(other.path))
        return offset

    @property
    def start(self):
        return min(t for t in (self.video_offset, self.audio_offset) if t is not None)

    @property
    def end(self):
        return min(t for t in (self.video_offset, self.audio_offset) if t is not None) + self.duration

    @staticmethod
    def get_creation_time(obj):
        creation_time = obj["format"]["tags"]["creation_time"]
        if creation_time is not None:
            return parser.parse(creation_time)
        return creation_time

    @staticmethod
    def get_duration(obj):
        return float(obj["format"]["duration"])

    @staticmethod
    def get_video_offset(obj):
        for stream in obj["streams"]:
            if stream["codec_type"] == "video":
                return float(stream["start_time"])
        return None

    @staticmethod
    def get_audio_offset(obj):
        for stream in obj["streams"]:
            if stream["codec_type"] == "audio":
                return float(stream["start_time"])
        return None

    def overlaps(self, other: "Media", padding=0., offset=0., after=0.) -> bool:
        if after is None:
            after = 0
        start = (self.creation_time - datetime.timedelta(seconds=padding))
        end = self.creation_time + datetime.timedelta(seconds=self.duration + padding + after)
        other_start = other.creation_time + datetime.timedelta(seconds=offset)
        other_end = other.creation_time + datetime.timedelta(seconds=other.duration + padding + offset)
        return not (other_end < start or end < other_start)

    def time_overlaps(self, t, padding=0.) -> bool:
        start = self.creation_time - datetime.timedelta(seconds=padding)
        end = self.creation_time + datetime.timedelta(seconds=self.duration) + datetime.timedelta(seconds=padding)
        return (start <= t <= end)

    def induce(self, sequence: "Sequence", name=None, padding=0., offset=0., after=None):
        return Sequence(name, [media for media in sequence if self.overlaps(media, padding=padding, offset=offset, after=after)])

    @staticmethod
    def intersection_size(a, b):
        end_a = a.creation_time + datetime.timedelta(seconds=a.duration)
        end_b = b.creation_time + datetime.timedelta(seconds=b.duration)
        return min(end_a, end_b) - max(a.creation_time, b.creation_time)

    def closest(self, sequence: "Sequence") -> Optional["Media"]:
        if len(sequence.elements) == 0: return None
        return max(sequence.elements, key=lambda x: Media.intersection_size(self, x))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path.name})"

class Sequence:
    def __init__(self, name: str, elements = None):
        self.name = name
        self.elements = elements if elements else []

    def append(self, *args, **kwargs):
        self.elements.append(*args, **kwargs)

    def sort(self, *args, **kwargs):
        self.elements.sort(*args, **kwargs)

    def filter(self, *args, **kwargs):
        self.elements.filter(*args, **kwargs)

    def __get__(self, *args, **kwargs):
        return self.elements.__get__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return iter(self.elements)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {repr(self.elements)})"

class Sequences:
    def __init__(self, org):
        self.org = org

    @staticmethod
    def from_directory(path: pathlib.Path):
        org = {}
        for p in pathlib.Path(path).glob("**/*.mk*"):
            camera = p.name.split(".")[-2]
            if camera not in org:
                org[camera] = Sequence(camera)
            try:
                org[camera].append(Media(p))
            except Exception:
                pass
        for _, value in org.items():
            value.elements.sort(key=lambda m: m.creation_time)
        return Sequences(org)

    def items(self, *args, **kwargs):
        return self.org.items(*args, **kwargs)

class OffsetGraph:
    def __init__(self):
        self.adjacency_list = {}

    def compute_start(self, target, media, offset, after):
        start = max(0, target.start - (media.start + offset["time_offset"]))
        return start

    def start(self, target, sequence, after):
        closest = target.closest(sequence)
        offset = self[target, closest]
        return [self.compute_start(target, media, offset, after) for media in sequence]

    def offset(self, target, sequence):
        closest = target.closest(sequence)
        offset = self[target, closest]
        return offset["time_offset"]

    def compute_end(self, target, media, offset, after):
        if after is None:
            after = 0
        end = min((target.start + max(target.duration, after)) - (media.start + offset["time_offset"]), media.duration)
        return end

    def end(self, target, sequence, after):
        closest = target.closest(sequence)
        offset = self[target, closest]
        return [self.compute_end(target, media, offset, after) for media in sequence]

    def __getitem__(self, index):
        if isinstance(index, slice):
            assert False
        elif isinstance(index, tuple):
            return self.adjacency_list[index[0]][index[1]]
        return self.adjacency_list[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            assert False
        elif isinstance(index, tuple):
            if index[0] not in self.adjacency_list:
                self.adjacency_list[index[0]] = {}
            if index[1] not in self.adjacency_list:
                self.adjacency_list[index[1]] = {}
            self.adjacency_list[index[0]][index[1]] = value
            self.adjacency_list[index[1]][index[0]] = value
        else:
            self.adjacency_list[index] = value


def setup(target, sequences, after=None, padding=0., threshold=4):

    # find overlaps
    _overlaps = Sequences({name: target.induce(sequence, name=name, padding=padding, after=after) for name, sequence in sequences.items()})
    # find relations
    _relations = OffsetGraph()
    for name, sequence in _overlaps.items():
        closest_media = target.closest(sequence)
        _relations[target, closest_media] = target.offset(closest_media)
    # find overlaps with complete offset information
    _overlaps = Sequences({
        name: target.induce(sequence, name=name, padding=0., offset=_relations[target, target.closest(sequence)]["time_offset"], after=after)
        for name, sequence in _overlaps.items()
    })
    return _overlaps, _relations

def _sync(target, _overlaps, _relations, after, script=True, output=None):
    # generate script
    return generate_sync_script(target, _overlaps, _relations, after, script, output)

def generate_sync_script(target, sequences, relations, after, script=True, output=None):
    complex = FilterComplex()
    movies, concat_filters = zip(*generate_concatenation_filters(target, complex, sequences, relations, after, script))
    video_streams = []
    output_streams = []
    for concat, (name, sequence) in zip(concat_filters, sequences.items()):
        offset = relations.offset(target, sequence)
        if offset == 0:
            pts = "PTS-STARTPTS"
        else:
            pts = f"PTS{'' if offset >= 0 else '+'}{offset}/TB"
        if concat.video_stream:
            setpts = SetPTS(f"setpts_concat_{name}", input_pads=[concat.video_stream], expr=pts)
            complex << setpts
            video_streams.append(setpts.video_stream)
        if concat.audio_stream:
            asetpts = ASetPTS(f"asetpts_concat_{name}", input_pads=[concat.audio_stream], expr=pts)
            output_streams.append(asetpts)
            complex << asetpts

    xstack = XStack("stack", input_pads=video_streams, shortest=0, inputs=len(video_streams))

    complex << xstack
    response = {}
    if script:
        response["input_flags"] = [movie.ffmpeg_str() for _movies in movies for movie in _movies]
        output_streams.sort(key=lambda x: target.name in x.name, reverse=True)
        response["output_flags"] = [
            f"-map '[{output_stream.name}]'" for output_stream in output_streams
        ]
        response["output_flags"].append(f"-map '[{xstack.name}]' -c:v libx264 -framerate 15 -crf 15")
        if output:
            response["output_flags"].append(f"-f matroska {output}")
    else:
        null = NullSink(f"null_{xstack.name}", input_pads=[xstack.video_stream])
        complex << null
        for output_stream in output_streams:
            null = ANullSink(f"null_{output_stream.name}", input_pads=[output_stream.audio_stream])
            complex << null
    response["filter_complex"] = complex.ffmpeg_str(newline=True)
    return response

def generate_concatenation_filters(target, complex, sequences, relations, after, script=True):
    input_counter = c_int64(0)
    return [
        generate_concatenation_filter(target, complex, name, sequence, relations, after, input_counter, script)
        for name, sequence in sequences.items()
    ]

def generate_concatenation_filter(target, complex, name, sequence, relations, after, input_counter, script=True):
    movies, audio_chains, video_chains = merge_sequence(target, complex, name, sequence, relations, after, input_counter, script)
    inputs = []
    if len(video_chains) and len(audio_chains):
        for video, audio in zip(video_chains, audio_chains):
            inputs.append(video)
            inputs.append(audio)
        concat = Concat(f"{name}_cat", inputs, n=len(video_chains), v=1, a=1)
    else:
        for audio in audio_chains:
            inputs.append(audio)
        concat = Concat(f"{name}_cat", inputs, n=len(audio_chains), v=0, a=1)
    complex << concat
    return movies, concat

def merge_sequence(target, complex, name, sequence, relations, after, input_counter, script=True):
    movies = []
    audio_chains = []
    video_chains = []
    starts, ends = relations.start(target, sequence, after), relations.end(target, sequence, after)
    for i, media in enumerate(sequence):
        if starts[i] > media.duration or ends[i] < 0:
            print(starts[i], ends[i], media.duration)
            continue
        if script:
            movie = DummyMovie(f"{input_counter.value}", filename=media.path)
        else:
            movie = Movie(f"{name}_{i}", filename=media.path)
            complex << movie
        input_counter.value += 1
        movies.append(movie)

        trim_kwargs = {}
        if starts[i] != 0:
            trim_kwargs["start"] = starts[i]
        if ends[i] != media.duration:
            trim_kwargs["end"] = ends[i]

        if movie.audio_stream:
            if trim_kwargs:
                atrim = ATrim(f"atrim_{name}_{i}", **trim_kwargs)
                achain = movie.audio_stream << atrim
                audio_chains.append(achain)
                complex << achain
            else:
                audio_chains.append(movie.audio_stream)

        if movie.video_stream:
            if trim_kwargs:
                trim = Trim(f"trim_{name}_{i}", **trim_kwargs)
                vchain = movie.video_stream << trim
                video_chains.append(vchain)
                complex << vchain
            else:
                video_chains.append(movie.video_stream)
    return movies, audio_chains, video_chains


@click.group()
def cli():
    pass


@cli.command()
@click.option("--target", "-t", type=str, required=True, help="name of device")
@click.option("--directory", "-d", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--log-level", "-l", default=logging.INFO, type=LogLevel())
def presync(target, directory, log_level):
    conn = sqlite3.connect("database.sql",detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    setup_logging(level=log_level)
    sequences = Sequences.from_directory(pathlib.Path(directory))

    media_data = []
    media_2_id = {}
    media_id = 0
    for name, sequence in sequences.items():
        for media in sequence:
            media_data.append({
                "media_id": media_id,
                "device": name,
                "creation_time": media.creation_time,
                "duration": media.duration,
                "path": str(media.path)
            })
            media_2_id[media] = media_id
            media_id += 1
    df_media = pd.DataFrame(media_data).set_index("media_id")

    relation_data = []
    relation_id = 0
    for media in sequences.org[target]:
        for name, sequence in sequences.items():
            if name == target: continue
            closest = media.closest(sequence)
            target_record_start = media.path.name.split(".")[0]
            other_record_start = closest.path.name.split(".")[0]
            if target_record_start != other_record_start: continue
            offset = media.offset(closest)
            relation_data.append({
                "relation_id": relation_id,
                "target_media_id": media_2_id[media],
                "other_media_id": media_2_id[closest],
                "time_offset": offset["time_offset"],
                "frame_offset": offset["frame_offset"],
                "standard_score": offset["standard_score"],
            })
            relation_id += 1
    df_relations = pd.DataFrame(relation_data).set_index("relation_id")
    df_media.to_csv("media.csv")
    df_relations.to_csv("relations.csv")
    df_media.to_sql("media", conn)
    df_relations.to_sql("relation", conn)


    ### find overlaps
    ##_overlaps = Sequences({name: target.induce(sequence, name=name, padding=padding, after=after) for name, sequence in sequences.items()})
    ### find relations
    ##_relations = OffsetGraph()
    ##for name, sequence in _overlaps.items():
    ##    print(name)
    ##    closest_media = target.closest(sequence)
    ##    _relations[target, closest_media] = target.offset(closest_media)
    ### find overlaps with complete offset information
    ##_overlaps = Sequences({
    ##    name: target.induce(sequence, name=name, padding=0., offset=_relations[target, target.closest(sequence)]["time_offset"], after=after)
    ##    for name, sequence in _overlaps.items()
    ##})
    ##return _overlaps, _relations

@cli.command()
@click.option("--target", "-t", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--directory", "-d", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--output", "-o", type=str, required=True, help="output")
@click.option("--after", "-a", type=float, required=False, default=None, help="output")
@click.option("--log-level", "-l", default=logging.INFO, type=LogLevel())
def sync_graph(target, directory, output, after, log_level):
    setup_logging(level=log_level)
    sequences = Sequences.from_directory(pathlib.Path(directory))
    target = Media(target)
    overlaps, relations = setup(target, sequences, padding=3, after=after)

    parts = _sync(target, overlaps, relations, after=after, script=False)

    subprocess.run(["graph2dot", "-o", "graph.tmp"], input=parts["filter_complex"].encode())
    subprocess.run(["dot", "-Tpng", "graph.tmp", "-o", output])

@cli.command()
@click.option("--target", "-t", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--directory", "-d", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--output", "-o", type=str, required=True, help="output")
@click.option("--after", "-a", type=float, required=False, default=None, help="output")
@click.option("--log-level", "-l", default=logging.INFO, type=LogLevel())
def sync_script(target, directory, output, after, log_level):
    setup_logging(level=log_level)
    sequences = Sequences.from_directory(pathlib.Path(directory))
    target = Media(target)
    overlaps, relations = setup(target, sequences, padding=3, after=after)

    parts = _sync(target, overlaps, relations, after=after, script=True, output=output)

    command = (
        ["ffmpeg"]
        + parts["input_flags"]
        + ["-filter_complex", f'"{parts["filter_complex"]}"']
        + parts["output_flags"]
    )
    click.echo("\n".join(command))


@cli.command()
@click.option("--target", "-t", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--directory", "-d", type=click.Path(exists=True), required=True, help="directory containing videos")
@click.option("--output", "-o", type=str, required=True, help="output")
@click.option("--after", "-a", type=float, required=False, default=None, help="output")
@click.option("--log-level", "-l", default=logging.INFO, type=LogLevel())
def sync(target, directory, output, after, log_level):
    setup_logging(level=log_level)
    sequences = Sequences.from_directory(pathlib.Path(directory))
    target = Media(target)
    overlaps, relations = setup(target, sequences, padding=3, after=after)

    parts = _sync(target, overlaps, relations, after=after, script=True, output=output)

    command = (
        ["ffmpeg"]
        + [flag_part for flag in parts["input_flags"] for flag_part in flag.split(" ")]
        + ["-filter_complex", parts["filter_complex"]]
        + [flag_part.strip("'") for flag in parts["output_flags"] for flag_part in flag.split(" ")]
    )
    subprocess.run(command)

if __name__ == "__main__":
    cli()
