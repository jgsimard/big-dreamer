import os
from typing import Iterable, List

import cv2
import numpy as np
import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from torch.nn import Module
from torch import Tensor
from torch import nn


def cat(x: Tensor, y: Tensor) -> Tensor:
    """
    Concatenate x and y along the channel dimension

    :param x:
    :param y:
    :return:
    """
    return torch.cat([x, y], dim=1)


def chunk(x: Tensor) -> List[Tensor]:
    """
    chunk x in two along the channel dimension

    :param x:
    :return:
    """
    return torch.chunk(x, 2, dim=1)


def stack(x: List[Tensor]) -> Tensor:
    """
    stack list of tensor while skiping the first element

    :param x:
    :return:
    """
    return torch.stack(x[1:], dim=0)


def prefill(t: int) -> List[Tensor]:
    """
    stuff

    :param t:
    :return:
    """
    return [torch.empty(0)] * t


def polyak(target_param: Tensor, param: Tensor, weight: float):
    """
    Polyak averaging for ONE parameter (soft update)

    :param target_param:
    :param param:
    :param weight:
    :return:
    """
    target_param.data.copy_(param.data * weight + target_param.data * (1.0 - weight))


def polyak_update(target: nn.Module, base: nn.Module, weight: float):
    """
    Perform polyack averaging (soft update) for a nn.Module

    :param target:
    :param base:
    :param weight:
    :return:
    """
    for target_param, param in zip(target.parameters(), base.parameters()):
        polyak(target_param, param, weight)

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
device = None


def init_gpu(use_gpu=True, gpu_id=0):
    """
    Initialize the GPU.

    :param use_gpu: if True, use GPU.
    :param gpu_id: id of the GPU to use.
    :return: None
    """

    global device  # pylint: disable=global-statement

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


def preprocess_observation_(observation, bit_depth) -> None:
    """
    Preprocesses an observation inplace.
    (from float32 Tensor [0, 255] to [-0.5, 0.5])

    Args:
        observation: the observation to preprocess.
        bit_depth: the bit depth of the observation

    Returns:
        None

    """

    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2**bit_depth).sub_(0.5)
    # Dequantise:
    # (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2**bit_depth))


def postprocess_observation(observation, bit_depth) -> np.ndarray:
    """
    Postprocess an observation for storage.
    (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])

    Args:
        observation: observation to process
        bit_depth: bit depth to quantise to

    Returns:
        np.ndarray: postprocessed observation
    """

    return np.clip(
        np.floor((observation + 0.5) * 2**bit_depth) * 2 ** (8 - bit_depth),
        0,
        2**8 - 1,
    ).astype(np.uint8)


def images_to_observation(images, bit_depth, observation_shape) -> np.ndarray:
    """
    Converts a list of images to a single observation.

    Args:
        images: list of images
        bit_depth: bit depth of the images
        observation_shape: shape of the observation

    Returns:
        observation: observation
    """

    # Resize and put channel first
    images = torch.tensor(
        cv2.resize(images, observation_shape, interpolation=cv2.INTER_LINEAR).transpose(
            2, 0, 1
        ),
        dtype=torch.float32,
    )

    # Quantise, centre and dequantise inplace
    preprocess_observation_(images, bit_depth)

    # Add batch dimension
    return images.unsqueeze(dim=0)


def build_mlp(
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        activation: str = 'ELU',
        output_activation: str = 'Identity',
) -> nn.Sequential:
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers
                                + the output layer
    """
    if isinstance(activation, str):
        activation = getattr(nn, activation)
    if isinstance(output_activation, str):
        output_activation = getattr(nn, output_activation)

    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation())
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation())
    return nn.Sequential(*layers)
