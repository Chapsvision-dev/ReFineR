import torch
from typing import Union, List, Dict, Tuple


class NamedTensor:
    """
    Helper class for multichannel tensors using pytorch
    """

    @staticmethod
    def _format_to_tensor(
        tensor: Union[torch.Tensor, List[torch.Tensor]], channel_dim: int
    ) -> torch.Tensor:
        """
        Tensors must all have shape ... or ... x N
        """
        if isinstance(tensor, list):
            tensor = torch.concatenate(tensor, dim=channel_dim)
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"tensor should be of type Tensor or list[Tensor] not {type(tensor)}"
            )
        return tensor

    def __init__(
        self,
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        channel_names: List[str],
        channel_dim: int,
    ):
        """
        tensor: ... x N x ..., Internally, the channel dimension is 0
        channel_names: list[string] names of each channel
        channel_dim: int index of the channel
        """
        self.channel_dim = channel_dim
        self._tensor = self._format_to_tensor(tensor, self.channel_dim).moveaxis(
            self.channel_dim, 0
        )
        c = len(channel_names)
        if c != self._tensor.shape[0]:
            raise ValueError(
                f"Number of channels should match ({self._tensor.shape[0], c})"
            )
        self._names_dict = {channel_names[i]: i for i in range(len(channel_names))}

    def clone(self):
        return NamedTensor(torch.clone(self[self.names]), self.names, self.channel_dim)

    @property
    def tensor(self) -> torch.Tensor:
        """
        Public tensor keeps its channel dimension while private _tensor has channel dimension to 0
        """
        return self._tensor.moveaxis(0, self.channel_dim)

    @tensor.setter
    def tensor(self, t: torch.Tensor):
        cn = t.shape[self.channel_dim]
        c = len(self)
        if cn != c:
            raise ValueError(
                f"Cannot set variable tensor as the new number of channels {cn} doesn't match the existing one {c}"
            )
        self._tensor = t.moveaxis(self.channel_dim, 0)

    @property
    def names(self) -> List[str]:
        return list(self._names_dict.keys())

    def reshape(self, new_size: Tuple["int"], new_channel_dim: int):
        t = self.tensor.reshape(*new_size)
        if t.shape[new_channel_dim] != self._tensor.shape[0]:
            raise ValueError("Cannot reshape to a different channel dimension")
        self._tensor = t.moveaxis(new_channel_dim, 0)
        self.channel_dim = new_channel_dim

    def subtensor(self, index: Union[str, List[str]]):
        if isinstance(index, str):
            index = [index]
        for c in index:
            if c not in self.names:
                raise ValueError(f"Unknown channel {c}")
        return NamedTensor(self[index], index, self.channel_dim)

    def remove_channels(self, index: Union[str, List[str]]):
        if isinstance(index, str):
            index = [index]
        for c in index:
            if c not in self.names:
                raise ValueError(f"Unknown channel {c}")
        new_names = self.unary_names(index)
        self._tensor = self[new_names].moveaxis(self.channel_dim, 0)
        self._names_dict = {new_names[i]: i for i in range(len(new_names))}

    def rename_channels(self, name_map: Dict[str, str]):
        """
        rename the channels of the tensor
        name_map: previous_name: new_name
        """
        for n in name_map:
            if n not in self.names:
                raise ValueError(f"name {n} not a channel in tensor")
            if name_map[n] in self.names:
                raise ValueError(f"name {name_map[n]} already a channel name in tensor")
            self._names_dict[name_map[n]] = self._names_dict.pop(n)

    @property
    def shape_other(self) -> Tuple[int, ...]:
        return self._tensor.shape[1:]

    def __getitem__(self, index: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(index, list):
            if len(index) == 0:
                zero_shape = list(self.tensor.shape)
                zero_shape[self.channel_dim] = 0
                return torch.zeros(
                    *zero_shape, device=self._tensor.device, dtype=self._tensor.dtype
                )  # ... x 0 x ...
            indices = []
            for c in index:
                if c not in self.names:
                    raise ValueError(f"Unknown channel {c}")
                indices.append(self._names_dict[c])
            indices = torch.tensor(indices, device=self._tensor.device)
            return self._tensor[indices].moveaxis(0, self.channel_dim)  # ... x N' x ...
        elif isinstance(index, str):
            indices = torch.tensor(self._names_dict[index], device=self._tensor.device)
            return self._tensor[indices]  # ... x ...
        else:
            raise TypeError(
                f"invalid index {index} of type {type(index)}, should be string or list of string"
            )

    def __setitem__(
        self,
        index: Union[str, List[str]],
        value: Union[torch.Tensor, List[torch.Tensor]],
    ):
        if not isinstance(index, str) and not isinstance(index, list):
            raise TypeError(
                f"invalid type {type(index)}, should be string or list of string"
            )

        if isinstance(index, str):
            index = [index]
            if isinstance(value, List):
                raise TypeError(
                    f"invalid type {type(value)}, a string key should be associated to a tensor"
                )
            value = value.unsqueeze(self.channel_dim)

        value = (
            NamedTensor._format_to_tensor(value, self.channel_dim)
            .moveaxis(self.channel_dim, 0)
            .to(self._tensor.dtype)
        )
        ci, cv = len(index), value.shape[0]
        if ci != cv:
            raise ValueError(
                f"Update channel number ({cv}) should match with given index size ({ci})"
            )

        to_add_names = []
        to_add_value_indices = []
        to_update_tensor_indices = []
        to_update_value_indices = []

        if isinstance(index, list):
            for i in range(len(index)):
                c = index[i]
                if c not in self.names:
                    to_add_names.append(c)
                    to_add_value_indices.append(i)
                else:
                    to_update_tensor_indices.append(self._names_dict[c])
                    to_update_value_indices.append(i)

        self._tensor[to_update_tensor_indices] = value[to_update_value_indices]
        self._add_channels(
            value[to_add_value_indices].moveaxis(0, self.channel_dim), to_add_names
        )

    def __contains__(self, item) -> bool:
        return item in self.names

    def unary_names(self, unary_indices: Union[str, List]) -> List[str]:
        names = []
        if isinstance(unary_indices, str):
            unary_indices = [unary_indices]
        for c in self.names:
            if c not in unary_indices:
                names.append(c)
        return names

    def __len__(self):
        return len(self._names_dict)

    @classmethod
    def fuse(cls, named_tensors: List):
        t = named_tensors[0]
        for i in range(1, len(named_tensors)):
            t._add_channels(named_tensors[i].tensor, named_tensors[i].names)
        return t

    def _add_channels(
        self,
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        channel_names: Union[str, List[str]],
    ):
        if len(channel_names) == 0:
            return

        if isinstance(channel_names, str):
            channel_names = [channel_names]
        for c in channel_names:
            if c in self._names_dict:
                raise ValueError(
                    f"Existing name {c} in the named tensor, use another one"
                )
        tensor = NamedTensor._format_to_tensor(tensor, self.channel_dim).moveaxis(
            self.channel_dim, 0
        )

        self._tensor = torch.concatenate((self._tensor, tensor), dim=0)
        n = len(self)
        names_dict = {channel_names[i]: n + i for i in range(len(channel_names))}
        self._names_dict.update(names_dict)
