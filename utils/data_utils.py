from torch_geometric.datasets import Planetoid, AttributedGraphDataset, Twitch, LastFMAsia
import os.path as osp


def data_loader(tgm_type, name, transform):
    path = osp.join('data', tgm_type)

    if tgm_type == "Planetoid":
        dataset = Planetoid(path, name=name, transform=transform)
    elif tgm_type == "AttributedGraphDataset":
        dataset = AttributedGraphDataset(path, name=name, transform=transform)
    elif tgm_type == "Twitch":
        dataset = Twitch(path, name=name, transform=transform)
    elif tgm_type == "LastFMAsia":
        dataset = LastFMAsia(path, transform=transform) # here there is no name argument
    else:
        raise NotImplementedError(f"Wrong torch_geometric type or not implemented error: {tgm_type}")

    return dataset
