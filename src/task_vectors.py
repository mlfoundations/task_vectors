import torch
from torch import Tensor
from abc import ABC, abstractmethod


class TaskVectorABC(ABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} is present in the pretrained state dict but not in the task vector")
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


class TaskVector(TaskVectorABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)


class TaskVectorTopKZero(TaskVectorABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, top_k: float = 0):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k = top_k
        for key, value in self.vector.items():
            self.vector[key] = self.mask(value)

    def mask(self, tensor: Tensor) -> Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.ones(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)

            return mask * tensor
