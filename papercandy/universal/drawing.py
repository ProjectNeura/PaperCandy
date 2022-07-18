import cv2 as _cv
from os import PathLike
from abc import abstractmethod
from typing import Any, Union
from typing_extensions import Self
from functools import singledispatch
from numpy import tan as _tan, sin as _sin, ones as _ones, uint8 as _uint8, ndarray as _ndarray

from papercandy import network as _network
from papercandy.universal import utils as _utils


class Drawer(object):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def show(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Union[str, PathLike]) -> Self:
        raise NotImplementedError


class NetworkDrawer(Drawer):
    def __init__(self, width: int, height: int, bg: Union[int, tuple[int]] = 255, margin_start: Union[int, float] = 0.2,
                 margin_top: Union[int, float] = 0.1, margin_end: Union[int, float] = 0.2,
                 margin_bottom: Union[int, float] = 0.1):
        self._width: int = width
        self._height: int = width
        self._margin_start: int = round(margin_start * width if -1 < margin_start < 1 else margin_start)
        self._margin_top: int = round(margin_top * height if -1 < margin_top < 1 else margin_top)
        self._margin_end: int = round(margin_end * width if -1 < margin_end < 1 else margin_end)
        self._margin_bottom: int = round(margin_bottom * height if -1 < margin_bottom < 1 else margin_bottom)
        self._display_width: int = width + self._margin_start + self._margin_end
        self._display_height: int = height + self._margin_top + self._margin_bottom
        self._canvas: _ndarray = _ones((self._display_height, self._display_width), dtype=_uint8) * bg \
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
    def _create_canvas(width: int, height: int, color: tuple[int]) -> _ndarray:
        canvas = _ones((height, width, len(color)), dtype=_uint8)
        canvas[:] = color
        return canvas

    def set_margin_start(self, margin_start: Union[int, float]) -> Self:
        self._margin_start: int = round(margin_start * self._width if -1 < margin_start < 1 else margin_start)
        return self

    def set_margin_top(self, margin_top: Union[int, float]) -> Self:
        self._margin_top: int = round(margin_top * self._height if -1 < margin_top < 1 else margin_top)
        return self

    def set_margin_end(self, margin_end: Union[int, float]) -> Self:
        self._margin_end: int = round(margin_end * self._width if -1 < margin_end < 1 else margin_end)
        return self

    def set_margin_bottom(self, margin_bottom: Union[int, float]) -> Self:
        self._margin_bottom: int = round(margin_bottom * self._height if -1 < margin_bottom < 1 else margin_bottom)
        return self

    def rev_y(self, y: int) -> int:
        return self._display_height - y

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, parent_width: int, offset_x: int, offset_y: int,
                  color: Union[int, tuple[int]] = 0) -> Self:
        offset_x, offset_y = offset_x + self._margin_start, offset_y + self._margin_bottom
        thickness = parent_width * 0.008
        thickness = round(thickness) if thickness > 1 else 1
        _cv.line(self._canvas, (round(x1 + offset_x), self.rev_y(round(y1 + offset_y))),
                 (round(x2 + offset_x), self.rev_y(y2 + offset_y)), color=color, thickness=thickness)
        return self

    def draw_text(self, text: str, parent_width: int, angle: int, offset_x: int, offset_y: int,
                  color: Union[int, tuple[int]] = 0) -> Self:
        font_size = parent_width * 0.024 / len(text)
        interval_x = round(parent_width / (len(text) + 1))
        interval_y = round(interval_x * _tan(_utils.angle2radian(angle)))
        offset_x, offset_y = offset_x + self._margin_start + interval_x, offset_y + self._margin_bottom + interval_y
        thickness = parent_width * 0.008
        thickness = round(thickness) if thickness > 1 else 1
        for c in text:
            _cv.putText(self._canvas, c, (offset_x, self.rev_y(offset_y)), fontFace=_cv.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_size, color=color, thickness=thickness)
            offset_x, offset_y = offset_x + interval_x, offset_y + interval_y
        return self

    @staticmethod
    def cal_bottom_line(layer_width: int, layer_angle: int) -> int:
        return round(layer_width * _sin(_utils.angle2radian(layer_angle)))

    def save(self, path: Union[str, PathLike]) -> Self:
        _cv.imwrite(path, _cv.cvtColor(self._canvas, _cv.COLOR_BGR2RGB))
        return self

    def show(self) -> Self:
        _cv.imshow("Network Structure", _cv.cvtColor(self._canvas, _cv.COLOR_BGR2RGB))
        _cv.waitKey(0)
        return self


@singledispatch
def draw(obj: Any, *args, **kwargs) -> Drawer:
    raise TypeError(f"No known case for type {type(obj)}.")


@draw.register(_network.LayerInfoList)
def _(lil: _network.LayerInfoList, interval: Union[int, float], color: Union[int, tuple[int]] = 0,
      bg: Union[int, tuple[int]] = 255) -> Drawer:
    if interval < 0:
        raise ValueError("`interval` cannot be negative.")
    drawer = NetworkDrawer(*lil(interval), bg=bg)
    offset_x, offset_y = 0, 0
    for layer in lil:
        drawer(layer.width, layer.g_width, layer.height, layer.angle, round(offset_x), round(offset_y), layer.name,
               description=layer.description, color=color)
        offset_x += layer.g_width + layer.parse_interval(interval)
    return drawer


@draw.register(_network.NetworkC)
def _(network: _network.NetworkC, interval: Union[int, float] = 0.2, color: Union[int, tuple[int]] = 0,
      bg: Union[int, tuple[int]] = 255) -> Drawer:
    if interval < 0:
        raise ValueError("`interval` cannot be negative.")
    return draw(network.structure(), interval, color, bg)
