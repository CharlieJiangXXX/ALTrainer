from dataclasses import dataclass
import typing


@dataclass
class AcquisitionBatch:
    """
    Wrapper for results of acquire_batch() describing a batch for acquisition.
    """

    indices: typing.List[int]
    scores: typing.List[float]
    original_scores: typing.Optional[typing.List[float]]
