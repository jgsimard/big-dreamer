import os
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
    """
    Plots min, max and mean + standard deviation bars of a population over time.

    :param xs: list of x values
    :param ys_population: list of y values
    :param title: title of the plot
    :param path: path to save the plot
    :param xaxis: x axis label
    """

    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    if isinstance(ys_population[0], (list, tuple)):
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
    """
    Writes a video from a list of frames.
    """

    frames = (
        np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255)
        .clip(0, 255)
        .astype(np.uint8)[:, :, :, ::-1]
    )  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape

    writer = cv2.VideoWriter(
        os.path.join(path, f"{title}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (W, H),
        True,
    )

    for frame in frames:
        writer.write(frame)
    writer.release()


class ActivateParameters:
    """
    Context manager to locally Activate the gradients.
    example:
    ```
    with ActivateParameters([module]):
        output_tensor = module(input_tensor)
    ```
    :param modules: iterable of modules. used to call .parameters() to freeze gradients.
    """

    def __init__(self, modules: Iterable[Module]):
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
    """
    Context manager to locally freeze gradients.
    In some cases with can speed up computation because gradients aren't calculated
    for these listed modules.
    example:
    ```
    with FreezeParameters([module]):
        output_tensor = module(input_tensor)
    ```
    :param modules: iterable of modules. used to call .parameters() to freeze gradients.
    """

    def __init__(self, modules: Iterable[Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


## from the homeworks


class classproperty:
    """
    Decorator to make a class property.
    """

    def __init__(self, f):
        """
        Decorator to enable access to properties of both classes and instances of classes.

        :param f:
        :returns:
        :rtype:
        """

        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    """
    Initialize the GPU.

    :param use_gpu: if True, use GPU.
    :param gpu_id: id of the GPU to use.
    :return: None
    """

    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print(f"Using GPU id {gpu_id}")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    """
    Set the device to use.
    """
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    """
    Convert a numpy array to a torch tensor.
    """

    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    """
    Convert a torch tensor to a numpy array.
    """

    return tensor.to("cpu").detach().numpy()


class Flatten(torch.nn.Module):
    """
    Flatten a tensor.
    """

    def forward(self, x):
        """
        Flatten a tensor.
        :param x: tensor to flatten.
        """

        batch_size = x.shape[0]
        return x.view(batch_size, -1)
