from dataclasses import dataclass

@dataclass
class PerturbationsConfig():
    inter_stroke: bool = False
    intra_stroke: bool = False
    intra_stroke_rev: bool = False
    stroke_order: bool = False

    def __post_init__(self):
        assert not self.inter_stroke or not self.stroke_order