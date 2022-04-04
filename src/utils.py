import inspect
import os
from functools import wraps
from typing import Iterable

import cv2
import numpy as np
import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from torch.nn import Module


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path="", xaxis="episode"):
    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = (
            ys.min(1),
            ys.max(1),
            ys.mean(1),
            ys.std(1),
            np.median(ys, 1),
        )
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(
            x=xs, y=ys_max, line=Line(color=max_colour, dash="dash"), name="Max"
        )
        trace_upper = Scatter(
            x=xs,
            y=ys_upper,
            line=Line(color=transparent),
            name="+1 Std. Dev.",
            showlegend=False,
        )
        trace_mean = Scatter(
            x=xs,
            y=ys_mean,
            fill="tonexty",
            fillcolor=std_colour,
            line=Line(color=mean_colour),
            name="Mean",
        )
        trace_lower = Scatter(
            x=xs,
            y=ys_lower,
            fill="tonexty",
            fillcolor=std_colour,
            line=Line(color=transparent),
            name="-1 Std. Dev.",
            showlegend=False,
        )
        trace_min = Scatter(
            x=xs, y=ys_min, line=Line(color=max_colour, dash="dash"), name="Min"
        )
        trace_median = Scatter(
            x=xs, y=ys_median, line=Line(color=max_colour), name="Median"
        )
        data = [
            trace_upper,
            trace_mean,
            trace_lower,
            trace_min,
            trace_max,
            trace_median,
        ]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
    plotly.offline.plot(
        {
            "data": data,
            "layout": dict(title=title, xaxis={"title": xaxis}, yaxis={"title": title}),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )


def write_video(frames, title, path=""):
    frames = (
        np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255)
        .clip(0, 255)
        .astype(np.uint8)[:, :, :, ::-1]
    )  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        os.path.join(path, "%s.mp4" % title),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (W, H),
        True,
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


class ActivateParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally Activate the gradients.
        example:
        ```
        with ActivateParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


## from the homeworks


def member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__:
    :returns:
    :rtype:

    """

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], defaults[index])

        wrapped__init__(self, *args, **kargs)

    return wrapper


def hidden_member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__:
    :returns:
    :rtype:

    """

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, "_" + name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, "_" + names[index]):
                    setattr(self, "_" + names[index], defaults[index])

        wrapped__init__(self, *args, **kargs)

    return wrapper


def tensor_member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__:
    :returns:
    :rtype:

    """
    import tensorflow as tf

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, tf.compat.v1.convert_to_tensor(arg))

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(
                        self,
                        names[index],
                        tf.compat.v1.convert_to_tensor(defaults[index]),
                    )
        wrapped__init__(self, *args, **kargs)

    return wrapper


class classproperty(object):
    def __init__(self, f):
        """Decorator to enable access to properties of both classes and instances of classes

        :param f:
        :returns:
        :rtype:

        """

        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
