import cv2 as _cv2
import numpy as _np
from os import PathLike
from abc import abstractmethod
from typing import Any, Union
from typing_extensions import Self
from functools import singledispatch
from matplotlib import pyplot as _plt

from papercandy import network as _network
from papercandy.universal import train as _train
from papercandy.universal import utils as _utils


class Drawer(object):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Self:
        """
        Draw.
        :param args: unknown
        :param kwargs: unknown
        :return: self
        """
        raise NotImplementedError

    @abstractmethod
    def show(self) -> Self:
        """
        Show the drawing.
        :return: self
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filename: Union[str, PathLike]) -> Self:
        """
        Save the drawing into a file.
        :param filename: filename
        :return: self
        """
        raise NotImplementedError


class NetworkDrawer(Drawer):
    def __init__(self, width: int, height: int, bg: Union[int, tuple[int]] = 255,
                 margin: Union[int, float, tuple, list] = (0.2, 0.1, 0.2, 0.2)):
        """
        Initialize the drawer.
        :param width: canvas width (pixels, without margin)
        :param height: canvas height (pixels, without margin)
        :param bg: background color (int for grayscale, tuple for multichannel)
        :param margin: margin (int for pixels, float for ratio)
            specific: horizontal, vertical / start, end, top, bottom
        """
        self._width: int = width
        self._height: int = width
        # parse margin
        if isinstance(margin, int):
            margin_start = margin_end = margin_top = margin_bottom = margin
        elif isinstance(margin, float):
            margin_start = margin_end = width * margin
            margin_top = margin_bottom = height * margin
        elif isinstance(margin, tuple) or isinstance(margin, list):
            if len(margin) == 2:
                margin_start = margin_end = margin[0]
                margin_top = margin_bottom = margin[1]
            elif len(margin) == 4:
                margin_start, margin_end, margin_top, margin_bottom = margin[0], margin[1], margin[2], margin[3]
            else:
                raise IndexError("Unexpected length of `margin`.")
        else:
            raise TypeError(f"No known case for `margin`: {type(margin)}.")
        self._margin_start: int = round(margin_start * width if -1 < margin_start < 1 else margin_start)
        self._margin_end: int = round(margin_end * width if -1 < margin_end < 1 else margin_end)
        self._margin_top: int = round(margin_top * height if -1 < margin_top < 1 else margin_top)
        self._margin_bottom: int = round(margin_bottom * height if -1 < margin_bottom < 1 else margin_bottom)
        # cal actual size
        self._display_width: int = width + self._margin_start + self._margin_end
        self._display_height: int = height + self._margin_top + self._margin_bottom
        # create canvas (grayscale or multichannel)
        self._canvas: _np.ndarray = _np.ones((self._display_height, self._display_width), dtype=_np.uint8) * bg \
            if isinstance(bg, int) else self._create_canvas(self._display_width, self._display_height, bg)

    def __call__(self, layer_width: int, graph_width: int, layer_height: int, layer_angle: int, offset_x: int,
                 offset_y: int, text: str, description: str = "", color: Union[int, tuple[int]] = 0) -> Self:
        self.draw_line(0, 0, 0, layer_height, graph_width, offset_x, offset_y, color)
        h = self.cal_bottom_line(layer_width, layer_angle)
        self.draw_line(0, 0, graph_width, h, graph_width, offset_x, offset_y, color) \
            .draw_line(graph_width, h, graph_width, h + layer_height, graph_width, offset_x, offset_y, color) \
            .draw_line(0, layer_height, graph_width, h + layer_height, graph_width, offset_x, offset_y, color)
        if description == "":
            self.draw_text(text, graph_width, layer_angle, offset_x, round(0.5 * layer_height), color)
        else:
            self.draw_text(text, graph_width, layer_angle, offset_x, round(0.6 * layer_height), color)\
                .draw_text(description, graph_width, layer_angle, offset_x, round(0.4 * layer_height), color)
        return self

    @staticmethod
    def _create_canvas(width: int, height: int, color: tuple[int]) -> _np.ndarray:
        """
        Create a canvas with multi channels.
        :param width: canvas width
        :param height: canvas height
        :param color: color
        :return: an array
        """
        canvas = _np.ones((height, width, len(color)), dtype=_np.uint8)
        canvas[:] = color
        return canvas

    def set_margin_start(self, margin_start: Union[int, float]) -> Self:
        self._margin_start: int = round(margin_start * self._width if -1 < margin_start < 1 else margin_start)
        return self

    def set_margin_end(self, margin_end: Union[int, float]) -> Self:
        self._margin_end: int = round(margin_end * self._width if -1 < margin_end < 1 else margin_end)
        return self

    def set_margin_top(self, margin_top: Union[int, float]) -> Self:
        self._margin_top: int = round(margin_top * self._height if -1 < margin_top < 1 else margin_top)
        return self

    def set_margin_bottom(self, margin_bottom: Union[int, float]) -> Self:
        self._margin_bottom: int = round(margin_bottom * self._height if -1 < margin_bottom < 1 else margin_bottom)
        return self

    def rev_y(self, y: int) -> int:
        """
        Reverse the y coordinate vertically.
        :param y: original y coordinate
        :return: reversed y coordinate
        """
        return self._display_height - y

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, parent_width: int, offset_x: int, offset_y: int,
                  color: Union[int, tuple[int]] = 0) -> Self:
        """
        Draw a line.
        :param x1: x coordinate of point 1
        :param y1: y coordinate of point 1
        :param x2: x coordinate of point 2
        :param y2: y coordinate of point 2
        :param parent_width: the parent area width which is used to calculate the relative thickness
        :param offset_x: holistic offset on the X axis
        :param offset_y: holistic offset on the Y axis
        :param color: line color (int for grayscale, tuple for multichannel)
            NOTICE: If the background is set as grayscale, the line color should be grayscale as well and vise versa.
        :return: self
        """
        offset_x, offset_y = offset_x + self._margin_start, offset_y + self._margin_bottom
        thickness = parent_width * 0.008
        thickness = round(thickness) if thickness > 1 else 1
        _cv2.line(self._canvas, (round(x1 + offset_x), self.rev_y(round(y1 + offset_y))),
                  (round(x2 + offset_x), self.rev_y(y2 + offset_y)), color=color, thickness=thickness)
        return self

    def draw_text(self, text: str, parent_width: int, angle: int, offset_x: int, offset_y: int,
                  color: Union[int, tuple[int]] = 0) -> Self:
        """
        Write a text.
        :param text: the text
        :param parent_width: the parent area width which is used to calculate the relative font size and thickness
        :param angle: the angle between the text direction and the right side of the X axis
        :param offset_x: holistic offset on the X axis
        :param offset_y: holistic offset on the Y axis
        :param color: text color (int for grayscale, tuple for multichannel)
            NOTICE: If the background is set as grayscale, the text color should be grayscale as well and vise versa.
        :return: self
        """
        font_size = parent_width * 0.024 / len(text)
        interval_x = round(parent_width / (len(text) + 1))
        interval_y = round(interval_x * _np.tan(_utils.angle2radian(angle)))
        offset_x, offset_y = offset_x + self._margin_start + interval_x, offset_y + self._margin_bottom + interval_y
        thickness = parent_width * 0.008
        thickness = round(thickness) if thickness > 1 else 1
        for c in text:
            _cv2.putText(self._canvas, c, (offset_x, self.rev_y(offset_y)), fontFace=_cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=font_size, color=color, thickness=thickness)
            offset_x, offset_y = offset_x + interval_x, offset_y + interval_y
        return self

    @staticmethod
    def cal_bottom_line(layer_width: int, layer_angle: int) -> int:
        """
        Calculate the extra height. See in the detailed documentation.
        :param layer_width: the layer width
        :param layer_angle: the angle between the bottom side and the right side of the X axis
        :return: the extra height
        """
        return round(layer_width * _np.sin(_utils.angle2radian(layer_angle)))

    def show(self, title: str = "Network Structure") -> Self:
        # WARNING: there might be some potential problem when the channels don't stand for RGB
        _cv2.imshow(title, _cv2.cvtColor(self._canvas, _cv2.COLOR_BGR2RGB))
        _cv2.waitKey(0)
        return self

    def save(self, filename: Union[str, PathLike]) -> Self:
        # WARNING: there might be some potential problem when the channels don't stand for RGB
        _cv2.imwrite(filename, _cv2.cvtColor(self._canvas, _cv2.COLOR_BGR2RGB))
        return self


class LossesDrawer(Drawer):
    def __init__(self, width: int, height: int, bg: str = "white"):
        self._width: int = round(width / 100)
        self._height: int = round(height / 100)
        self._losses: list[float] = []
        self._bg: str = bg

    def __call__(self, losses: list[float], color: str = "black") -> Self:
        self._losses += losses
        _plt.figure(figsize=(self._width, self._height))
        _plt.plot(_np.arange(1, len(self._losses) + 1), self._losses, marker="o", color=color, label="loss")
        _plt.xlabel("Epoch")
        _plt.ylabel("Loss")
        return self

    def show(self, title: str = "Training Loss") -> Self:
        _plt.title(title)
        _plt.show()
        return self

    def save(self, filename: Union[str, PathLike]) -> Self:
        _plt.savefig(filename)
        return self


@singledispatch
def draw(obj: Any, *args, **kwargs) -> Drawer:
    raise TypeError(f"No known case for type {type(obj)}, args: {args}, kwargs: {kwargs}.")


@draw.register(_network.LayerInfoList)
def _(lil: _network.LayerInfoList, interval: Union[int, float] = 0.1, color: Union[int, tuple[int]] = 0,
      bg: Union[int, tuple[int]] = 255, margin: Union[int, float, tuple, list] = (0.2, 0.1)) -> Drawer:
    drawer = NetworkDrawer(*lil(interval), bg, margin)
    offset_x, offset_y = 0, 0
    for layer in lil:
        drawer(layer.width, layer.g_width, layer.height, layer.angle, round(offset_x), round(offset_y), layer.name,
               layer.description, color)
        offset_x += layer.g_width + layer.parse_interval(interval)
    return drawer


@draw.register(_network.NetworkC)
def _(network: _network.NetworkC, interval: Union[int, float] = 0.1, color: Union[int, tuple[int]] = 0,
      bg: Union[int, tuple[int]] = 255, margin: Union[int, float, tuple, list] = (0.2, 0.1)) -> Drawer:
    return draw(network.structure(), interval, color, bg, margin)


@draw.register(_train.Trainer)
def _(trainer: _train.Trainer, width: int, height: int, color: str = "black", bg: str = "white") -> Drawer:
    drawer = LossesDrawer(width, height, bg)
    return drawer(trainer.losses, color)
