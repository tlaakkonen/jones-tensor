import numpy as np
import quimb.tensor as qtn
import autoray
from autoray import numpy as anp
from typing import Any
from .morselink import MorseLink, MorseCap, MorseCup, MorseX

def construct_matrices(qs: np.ndarray, batch: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.iscomplexobj(qs) and np.all(qs >= 0):
        As = np.sqrt(np.sqrt(qs))
    else:
        As = (qs + 1j*np.zeros_like(qs)) ** (1/4)

    z = np.zeros_like(As)

    M = np.stack([
        np.stack([z, 1j * As], axis=0),
        np.stack([-1j*As**-1, z], axis=0)
    ], axis=0)

    Rneg = np.stack([
        np.stack([As, z, z, z], axis=0),
        np.stack([z, z, As**-1, z], axis=0),
        np.stack([z, As**-1, As - As**-3, z], axis=0),
        np.stack([z, z, z, As], axis=0),
    ], axis=0).reshape(2, 2, 2, 2, -1)

    Rpos = np.stack([
        np.stack([As**-1, z, z, z], axis=0),
        np.stack([z, As**-1-As**3, As, z], axis=0),
        np.stack([z, As, z, z], axis=0),
        np.stack([z, z, z, As**-1], axis=0),
    ], axis=0).reshape(2, 2, 2, 2, -1)

    if not batch:
        As = As[..., 0]
        S = S[..., 0]
        Rpos = Rpos[..., 0]
        Rneg = Rneg[..., 0]

    return anp.array(As), anp.array(M), anp.array(Rpos), anp.array(Rneg)

def construct_network(qs: float | complex | np.ndarray, link: MorseLink) -> tuple[qtn.TensorNetwork, str | None]:
    """
    Construct a tensor network which when contracted will evaluate to the Jones polynomial of a link.

    NB: The contraction of this network is undefined when q = -1, you can either evaluate q slightly away 
    from -1, or compute |V(-1)| from its interpretation as the knot determinant.

    Parameters:
        qs: The value of q, or a 1D numpy array of q values for batching.
        link: A MorseLink presentation of the link L.

    Returns:
        net: A quimb.tensor.TensorNetwork representing the computation
        batch: If a batch of qs is provided, this is the batch index label, otherwise it is None. 
    """
    if isinstance(qs, (int, float, complex)) or (isinstance(qs, np.ndarray) and qs.shape == ()):
        qs = np.array([qs])
        batch = False
    elif isinstance(qs, np.ndarray) and len(qs.shape) == 1:
        batch = True
    else:
        raise ValueError("qs must be a scalar or 1D numpy array")
    
    As, M, Rpos, Rneg = construct_matrices(qs, batch)

    factor = (-As**3)**link.writhe() / (-As**-2-As**2)
    tensors = []
    strands = []
    nidx = 0
    for event in link.events:
        if isinstance(event, MorseCup):
            inds = [f"i{nidx}", f"i{nidx+1}"] + (["batch"] if batch else [])
            nidx += 2
            strands.insert(event.idx, inds[0])
            strands.insert(event.idx + 1, inds[1])
            data = M.copy()
            if factor is not None:
                data *= (factor[None, None, :] if batch else factor)
                factor = None
            tensors.append(qtn.Tensor(data=data, inds=tuple(inds), tags=["Cup"]))
        elif isinstance(event, MorseCap):
            inds = [strands[event.idx], strands[event.idx + 1]] + (["batch"] if batch else [])
            del strands[event.idx + 1]
            del strands[event.idx]
            tensors.append(qtn.Tensor(data=M.copy(), inds=tuple(inds), tags=["Cap"]))
        else:
            assert isinstance(event, MorseX)
            inds = [strands[event.idx], strands[event.idx + 1], f"i{nidx}", f"i{nidx+1}"] + (["batch"] if batch else [])
            nidx += 2
            strands[event.idx] = inds[2]
            strands[event.idx + 1] = inds[3]
            tensors.append(qtn.Tensor(data=Rpos.copy() if event.over else Rneg.copy(), inds=tuple(inds), tags=["X"]))

    net = qtn.TensorNetwork(tensors)
    return net, 'batch' if batch else None

def evaluate_jones(
    qs: float | complex | np.ndarray, link: MorseLink,
    optimize: str | Any = 'auto', batchsize: int = 128, 
    get: str | None = None, backend: str | None = None
) -> np.ndarray:
    """
    Evaluate the Jones polynomial V_L(q) of a link L at some values of q != -1.

    NB: The returned value is not well-defined for q = -1, you can either evaluate q slightly away from -1, 
    or compute |V(-1)| from its interpretation as the knot determinant.

    Parameters:
        qs: The value of q, or a 1D numpy array of q values for batching.
        link: A MorseLink presentation of the link L.
        optimize: The pathfinding strategy. See https://quimb.readthedocs.io/en/latest/tensor/tensor-contraction.html#the-optimize-kwarg.
        batchsize: The number of q values to compute simultaneously. Reduce this to reduce memory consumption.
        get: If None, return the value of the Jones polynomial. Set this to 'path-info' or 'tree' to get information about the contraction instead.
        backend: The autoray backend to compute with, if not already set. E.g: 'numpy' for CPU, 'cupy' for nvidia GPUs, 'mlx' for Apple M-series.
    """
    if isinstance(qs, np.ndarray) and len(qs.shape) == 1 and qs.shape[0] > batchsize:
        if get is not None: raise ValueError("non-None `get` is not compatible with batch splitting")
        res = np.zeros_like(qs)
        for idx in range(0, qs.shape[0], batchsize):
            res[idx:idx+batchsize] = evaluate_jones(qs[idx:idx+batchsize], link, optimize=optimize, batchsize=batchsize, get=get, backend=backend)
        return res
    
    if backend is not None:
        with autoray.backend_like(backend):
            net, batch = construct_network(qs, link)
    else:
        net, batch = construct_network(qs, link)

    output = net.contract(output_inds=(batch,) if batch is not None else (), optimize=optimize, backend=backend, get=get)

    if get is None:
        return autoray.to_numpy(output.data)
    else:
        return output
