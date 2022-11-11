import math
from enum import Enum
from typing import List, Optional, Set, Union

class StreamType(Enum):
    VIDEO = 1
    AUDIO = 2

class Stream:
    def __init__(self, name: str, type: StreamType):
        self.name = name
        self.type = type


    def __lshift__(self, other: Union["Filter", "FilterChain"]):
        if isinstance(other, Filter):
            other.input_pads.append(self)
        elif isinstance(other, FilterChain):
            other.input_pads.append(self)
        else:
            raise TypeError
        return other

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.name}:{self.type}>"

class FFMPegFlags:
    def __init__(self, name, **kwargs):
        self.name = name
        self.items = kwargs

    def flags(self) -> List[str]:
        return [element for key_value in self.items() for element in key_value]

class Filter:
    STREAM_TYPES = {StreamType.AUDIO, StreamType.VIDEO}

    def __init__(self, name: str, input_pads: Optional[List[Stream]] = None, outputs: Optional[List[Stream]] = None, **kwargs):
        self.name = name
        self.input_pads = input_pads if input_pads is not None else []
        self.outputs = outputs if input_pads is not None else self._generate_outputs()
        self.kwargs = kwargs



    @property
    def audio_stream(self) -> Optional[Stream]:
        try:
            return next(stream for stream in self.outputs if stream.type == StreamType.AUDIO)
        except StopIteration:
            return None

    @property
    def video_stream(self) -> Optional[Stream]:
        try:
            return next(stream for stream in self.outputs if stream.type == StreamType.VIDEO)
        except StopIteration:
            return None

    def _generate_outputs(self):
        return [
            Stream(f"{self.name}", stream_type)
            for stream_type in self.__class__.STREAM_TYPES
        ]

    def _input_links_str(self):
        if self.input_pads is None: return ""
        return "".join(f"[{_stream.name}]" for _stream in self.input_pads)

    def _output_links_str(self):
        return "".join(f"[{_stream.name}]" for _stream in self.outputs)

    def _filter_name_str(self):
        return self.__class__.FILTER_NAME

    def _filter_arguments_str(self):
        any_kwargs = "=" if self.kwargs else ""
        kwargs = ":".join(f"{self._gen_arg(key, value)}" for key, value in self.kwargs.items())
        return f"{any_kwargs}{kwargs}"

    def _gen_arg(self, key, value):
        if key == "expr":
            return value
        else:
            return f"{key}={value}"

    def ffmpeg_str(self, input_pads=True, outputs=True) -> str:
        input_links = self._input_links_str()
        filter_name = self._filter_name_str()
        filter_arguments = self._filter_arguments_str()
        output_links = self._output_links_str()
        return f"{input_links if input_pads else ''}{filter_name}{filter_arguments}{output_links if outputs else ''}"

    def resolve_stream_types(self) -> Set[StreamType]:
        types = self.__class__.STREAM_TYPES.copy()
        if self.input_pads:
            types = types.intersection(_stream.type for _stream in self.input_pads)
        if self.outputs:
            types = types.intersection(_stream.type for _stream in self.outputs)
        return types

    def __lshift__(self, other: Union["Filter", "FilterChain"]):
        if isinstance(other, Filter):
            other.input_pads += self.outputs
            return FilterChain(
                f"{self.name}_chain",
                [self, other],
                input_pads=self.input_pads
            )
        elif isinstance(other, FilterChain):
            other.input_pads += self.outputs
            other.filters.append(self)
            return other
        else:
            raise TypeError
        return other


class Concat(Filter):
    FILTER_NAME = "concat"
    STREAM_TYPES = {StreamType.AUDIO, StreamType.VIDEO}

    def __init__(self, name: str, input_pads: List["Filter"] = [], **kwargs):
        n_audio = kwargs.get("a", 0)
        n_video = kwargs.get("v", 0)
        n = kwargs["n"]
        voutputs = [Stream(f"{name}_video_{i}", StreamType.VIDEO) for i in range(n_video)]
        aoutputs = [Stream(f"{name}_audio_{i}", StreamType.AUDIO) for i in range(n_audio)]
        outputs = voutputs + aoutputs
        super(Concat, self).__init__(name, input_pads, outputs, **kwargs)

class ATrim(Filter):
    FILTER_NAME = "atrim"
    STREAM_TYPES = {StreamType.AUDIO}

    def __init__(self, *args, **kwargs):
        super(ATrim, self).__init__(*args, **kwargs)

class ASetPTS(Filter):
    FILTER_NAME = "asetpts"
    STREAM_TYPES = {StreamType.AUDIO}

    def __init__(self, *args, **kwargs):
        super(ASetPTS, self).__init__(*args, **kwargs)

class Trim(Filter):
    FILTER_NAME = "trim"
    STREAM_TYPES = {StreamType.VIDEO}

    def __init__(self, *args, **kwargs):
        super(Trim, self).__init__(*args, **kwargs)

class SetPTS(Filter):
    FILTER_NAME = "setpts"
    STREAM_TYPES = {StreamType.VIDEO}

    def __init__(self, *args, **kwargs):
        super(SetPTS, self).__init__(*args, **kwargs)

class XStack(Filter):
    FILTER_NAME = "xstack"
    STREAM_TYPES = {StreamType.VIDEO}

    def __init__(self, name: str, input_pads: List[Filter] = [], outputs: List[Filter] = [], **kwargs):
        assert kwargs.setdefault('input_pads', len(input_pads)) == len(input_pads)
        if not outputs:
            outputs = [Stream(name, StreamType.VIDEO)]
        if "layout" not in kwargs:
            kwargs["layout"] = XStack.layout(input_pads)
        if "input_pads" in kwargs:
            del kwargs["input_pads"]
        if "outputs" in kwargs:
            del kwargs["outputs"]
        super(XStack, self).__init__(name, input_pads=input_pads, outputs=outputs, **kwargs)

    @staticmethod
    def layout(streams: List[Stream]):
        assert all(stream.type == StreamType.VIDEO for stream in streams)
        dimensions = math.ceil(len(streams) * 0.5)
        parts = []
        for i, stream in enumerate(streams):
            col = i % dimensions
            row = i // dimensions
            if col > 0:
                col = f"w{col - 1}"
            if row > 0:
                row = f"h{row - 1}"
            parts.append(f"{col}_{row}")
        return "|".join(parts)

class DummyMovie(Filter):
    FILTER_NAME = "movie"
    AUDIO_VIDEO_SUFFIXES = ("mp4", "mkv")
    AUDIO_SUFFIXES = ("mp3", "mka")
    STREAM_TYPES = {StreamType.AUDIO, StreamType.VIDEO}

    def __init__(self, name: str, **kwargs):
        suffix = str(kwargs["filename"]).split(".")[-1]
        if suffix in DummyMovie.AUDIO_VIDEO_SUFFIXES:
            outputs = [
                Stream(f"{name}:v", StreamType.VIDEO),
                Stream(f"{name}:a", StreamType.AUDIO)
            ]
            kwargs["s"] = "dv+da"
        elif suffix in DummyMovie.AUDIO_SUFFIXES:
            outputs = [
                Stream(f"{name}:a", StreamType.AUDIO)
            ]
            kwargs["s"] = "da"
        super(DummyMovie, self).__init__(name, input_pads=[], outputs=outputs, **kwargs)

    def ffmpeg_str(self) -> str:
        return f"-i {self.kwargs['filename']}"

    @property
    def audio_stream(self) -> Optional[Stream]:
        try:
            return next(stream for stream in self.outputs if stream.type == StreamType.AUDIO)
        except StopIteration:
            return None

    @property
    def video_stream(self) -> Optional[Stream]:
        try:
            return next(stream for stream in self.outputs if stream.type == StreamType.VIDEO)
        except StopIteration:
            return None

class Movie(Filter):
    FILTER_NAME = "movie"
    AUDIO_VIDEO_SUFFIXES = ("mp4", "mkv")
    AUDIO_SUFFIXES = ("mp3", "mka")
    STREAM_TYPES = {StreamType.AUDIO, StreamType.VIDEO}

    def __init__(self, name: str, **kwargs):
        suffix = str(kwargs["filename"]).split(".")[-1]
        if suffix in Movie.AUDIO_VIDEO_SUFFIXES:
            outputs = [
                Stream(f"{name}_video", StreamType.VIDEO),
                Stream(f"{name}_audio", StreamType.AUDIO)
            ]
            kwargs["s"] = "dv+da"
        elif suffix in Movie.AUDIO_SUFFIXES:
            outputs = [
                Stream(f"{name}_audio", StreamType.AUDIO)
            ]
            kwargs["s"] = "da"
        super(Movie, self).__init__(name, input_pads=[], outputs=outputs, **kwargs)

    @property
    def audio_stream(self) -> Optional[Stream]:
        try:
            return next(stream for stream in self.outputs if stream.type == StreamType.AUDIO)
        except StopIteration:
            return None

    @property
    def video_stream(self) -> Optional[Stream]:
        try:
            return next(stream for stream in self.outputs if stream.type == StreamType.VIDEO)
        except StopIteration:
            return None

class ANullSink(Filter):
    FILTER_NAME = "anullsink"
    STREAM_TYPES = {StreamType.AUDIO}

    def __init__(self, *args, **kwargs):
        super(ANullSink, self).__init__(*args, **kwargs)
        self.outputs = []

class NullSink(Filter):
    FILTER_NAME = "nullsink"
    STREAM_TYPES = {StreamType.VIDEO}

    def __init__(self, *args, **kwargs):
        super(NullSink, self).__init__(*args, **kwargs)
        self.outputs = []


class FilterChain:

    def __init__(self,
                 name: str,
                 filters: List[Filter],
                 input_pads: Optional[List[Stream]] = None,
                 outputs: Optional[List[Stream]] = None):
        self.name = name
        self.filters = filters
        self.filters[0].input_pads = input_pads if not self.filters[0].input_pads else self.filters[0].input_pads
        if not outputs and not self.filters[-1].outputs and "Null" not in filters[-1].__class__.__name__:
            self.filters[-1].outputs = self._generate_outputs()

    @property
    def input_pads(self):
        return self.filters[0].input_pads

    @property
    def outputs(self):
        return self.filters[-1].outputs

    def _generate_outputs(self):
        types = self.resolve_stream_types()
        assert len(types) == 1
        return [Stream(self.name, next(iter(types)))]

    def ffmpeg_str(self) -> str:
        if len(self.filters) == 1:
            return self.filters[0].ffmpeg_str()
        return ",".join(
            _filter.ffmpeg_str(
                input_pads = i == 0,
                outputs = False
            )
            for i, _filter in enumerate(self.filters)
        ) + self._output_links_str()


    def _output_links_str(self):
        return f"[{self.name}]"

    def resolve_stream_types(self) -> Set[StreamType]:
        types = {StreamType.AUDIO, StreamType.VIDEO}
        for _filter in self.filters:
            types = types.intersection(_filter.resolve_stream_types())
        return types

class FilterComplex:

    def __init__(self):
        self.filter_chains: List[FilterChain] = []

    def __lshift__(self, other: Union[Filter, FilterChain]):
        if isinstance(other, Filter):
            _other = FilterChain(other.name, [other])
        else:
            _other = other
        self.filter_chains.append(_other)
        return self

    def ffmpeg_str(self, newline=False) -> str:
        sep = ";" + ("\n" if newline else "")
        return sep.join(_filter_chain.ffmpeg_str() for _filter_chain in self.filter_chains)

    def flags(self, newline=False) -> List[str]:
        return ["-filter_complex", self.ffmpeg_str(newline)]

class FFMpeg:

    def __init__(self):
        ...

if __name__ == "__main__":
    v1 = Stream("v1", StreamType.VIDEO)
    v2 = Stream("v2", StreamType.VIDEO)
    v3 = Stream("v3", StreamType.VIDEO)
    a1 = Stream("a1", StreamType.AUDIO)
    a2 = Stream("a2", StreamType.AUDIO)
    concat = Concat("blah", [v1, a1, v2, a2], n=2, v=1, a=1)

    print(concat.ffmpeg_str())
    print(concat.outputs)

    xstack = XStack("stack", input_pads=[v1, v2, v3])
    print(xstack.ffmpeg_str())

    print("============")
    print("")
    print("============")

    atrim_ = ATrim("atrim", end=3.4637)
    asetpts_ = ASetPTS("apts", expr="PTS")
    print((atrim_ << asetpts_).ffmpeg_str())
    print(atrim_.ffmpeg_str(), asetpts_.ffmpeg_str(), asetpts_.input_pads)

    print("============")
    print("")
    print("============")

    m = Movie("movie", filename="sync0.mkv")
    atrim = ATrim("atrim", end=3.4637)
    asetpts = ASetPTS("apts", expr="PTS")

    vtrim = Trim("trim", end=3.4637)
    vsetpts = SetPTS("pts", expr="PTS")
    chain_1 = m.video_stream << vtrim << vsetpts
    chain_2 = m.audio_stream << atrim << asetpts

    fc = FilterComplex()
    (fc << chain_1 << chain_2)
    print(chain_1.input_pads)
    print(fc.ffmpeg_str(newline=True))
