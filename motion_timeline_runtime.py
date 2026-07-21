from dataclasses import dataclass
import numpy as np


@dataclass
class RuntimeKeyFrame:

    index: int

    image: np.ndarray


class RuntimeMotionTimeline:

    def __init__(self):

        self.frames = []

    def add(self, image):

        self.frames.append(

            RuntimeKeyFrame(

                len(self.frames),

                image

            )

        )

    def first(self):

        return self.frames[0]

    def last(self):

        return self.frames[-1]

    def count(self):

        return len(self.frames)

    def __iter__(self):

        return iter(self.frames)