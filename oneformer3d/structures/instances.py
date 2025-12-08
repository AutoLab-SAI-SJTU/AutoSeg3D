# ------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import itertools
from typing import Any, Dict, List, Tuple, Union
import torch
import numpy as np


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    # def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
    #     """
    #     Args:
    #         item: an index-like object and will be used to index all the fields.

    #     Returns:
    #         If `item` is a string, return the data in the corresponding field.
    #         Otherwise, returns an `Instances` where all fields are indexed by `item`.
    #     """
    #     if type(item) == int:
    #         if item >= len(self) or item < -len(self):
    #             raise IndexError("Instances index out of range!")
    #         else:
    #             item = slice(item, None, len(self))

    #     ret = Instances(self._image_size)
    #     for k, v in self._fields.items():
    #         ret.set(k, v[item])
    #     return ret
    def _check_bool_mask(self, mask: torch.BoolTensor) -> None:
        if mask.numel() != len(self):
            raise IndexError(
                f"Boolean mask length {mask.numel()} does not match Instances length {len(self)}"
            )

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor, torch.Tensor]) -> "Instances":
        if isinstance(item, int):  # single index
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range")
            item = slice(item, item + 1)  # keep length =1 slice for all fields
        elif isinstance(item, torch.BoolTensor):
            self._check_bool_mask(item)
        elif isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            self._check_bool_mask(item)

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret
    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                # 确保在第二维度进行拼接,因为第1维度是batchsize,第2维度是queries_num
                values = torch.cat(values, dim=0)
                # values = torch.cat(values,dim=1)
            elif isinstance(v0, list):
                # values = list(itertools.chain(*values))
                # 将嵌套的列表平铺
                if len(values[0]) == 0:
                # 如果第一个列表为空，直接返回第二个列表
                    values = values[1]
                else:
                    assert len(values[0]) == len(values[1]), "Lists must have the same length"
                    # flat_list = list(itertools.chain(*values))
                    # # 如果平铺后的所有元素都是 tensor，则拼接成一个 tensor
                    # if flat_list and isinstance(flat_list[0], torch.Tensor):
                    #     values = torch.cat(flat_list, dim=0)
                    #     # 要返回的是一个list
                    #     values = [values]
                    # else:
                    #     values = flat_list

                    cat_values = []
                    for i in range(len(values[0])):
                        # 检查是否都是张量
                        if isinstance(values[0][i], torch.Tensor) and isinstance(values[1][i], torch.Tensor):
                            # 两个都是张量
                            if values[0][i].numel() > 0 and values[1][i].numel() > 0:
                                # 两个张量都不为空，进行拼接
                                cat_values.append(torch.cat([values[0][i], values[1][i]], dim=0))
                            elif values[0][i].numel() == 0 and values[1][i].numel() == 0:
                                # 两个张量都为空，报错或返回特定值
                                raise ValueError(f"Both values[0][{i}] and values[1][{i}] are empty tensors.")
                            else:
                                # 一个为空，一个不为空，使用非空的那个
                                non_empty_tensor = values[0][i] if values[0][i].numel() > 0 else values[1][i]
                                cat_values.append(non_empty_tensor)
                        elif isinstance(values[0][i], torch.Tensor) or isinstance(values[1][i], torch.Tensor):
                            # 其中一个是张量，另一个不是张量
                            non_empty_tensor = values[0][i] if isinstance(values[0][i], torch.Tensor) and values[0][i].numel() > 0 else values[1][i]
                            cat_values.append(non_empty_tensor)
                        else:
                            # 两个都不是张量，根据需求处理
                            raise TypeError(f"values[0][{i}] or values[1][{i}] is not a tensor.")

                    # cat_values = [torch.cat([values[0][i], values[1][i]], dim=0) if isinstance(values[0][i], torch.Tensor) else values[0][i] + values[1][i] for i in range(len(values[0]))]
                    values = cat_values
                
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
