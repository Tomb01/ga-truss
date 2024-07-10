from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt

Node = Annotated[npt.NDArray[np.int32], Literal[8]]