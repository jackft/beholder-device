import sqlite3

from datetime import datetime, time
from typing import List, Union

class Interval:
    def __init__(self, start: Union[str, datetime], end: Union[str, datetime]):
        if isinstance(start, str):
            self.start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")
        else:
            self.start = start
        if isinstance(end, str):
            self.end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S.%f")
        else:
            self.end = end

    def point_overlaps(self, date: Union[datetime, time]) -> bool:
        if isinstance(date, time):
            return self.start.time() <= date and date <= self.end.time()
        else:
            return self.start <= date and date <= self.end

class IntervalCollection:
    """This could be an interval tree, but that'd be premature"""
    def __init__(self, intervals: List[Interval]):
        self.intervals = intervals

    def point_overlaps(self, date: datetime) -> bool:
        return any(interval.point_overlaps(date) for interval in self.intervals)

    @staticmethod
    def from_sql(cur: sqlite3.Cursor):
        query = "SELECT start, end FROM blackout_interval"
        return IntervalCollection([Interval(row[0], row[1]) for row in cur.execute(query)])


class TimeInterval:
    def __init__(self, start: Union[str, datetime], end: Union[str, datetime]):
        if isinstance(start, str):
            self.start = datetime.strptime(start, "%H:%M:%S.%f").time()
        else:
            self.start = start.time()
        if isinstance(end, str):
            self.end = datetime.strptime(end, "%H:%M:%S.%f").time()
        else:
            self.end = end.time()

    def point_overlaps(self, t: time):
        if self.start <= self.end:
            return self.start <= t <= self.end
        return self.start <= t or t <= self.end


class RecordTime:
    """This could be an interval tree, but that'd be premature"""
    def __init__(self, intervals: List[TimeInterval]):
        self.intervals = intervals

    def point_overlaps(self, t: time):
        return any(interval.point_overlaps(t) for interval in self.intervals)

    @staticmethod
    def from_sql(cur: sqlite3.Cursor):
        query = "SELECT start, end FROM recordtime WHERE activated"
        return RecordTime([TimeInterval(row[0], row[1]) for row in cur.execute(query)])
