from typing import Union, List, Dict, TypedDict, Optional, Any
import torch
from abc import ABC, abstractmethod
from ..misc import softmax, find_top_indexes, show_dict_compact

class BaseEncodeOutputDict(TypedDict):
    ids: torch.Tensor
    mask: torch.Tensor
    position_offsets: Optional[torch.Tensor]

class EncodeOutputDict:
    def __init__(self, **kwargs):
        self.data = BaseEncodeOutputDict(
            ids=kwargs.pop('ids'),
            mask=kwargs.pop('mask'),
            position_offsets=kwargs.pop('position_offsets', None)
        )
        self.additional_data = kwargs if kwargs != None else {}
        if self.data['position_offsets'] == None:
            del self.data['position_offsets']

    def __getitem__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        elif key in self.additional_data:
            return self.additional_data[key]
        raise KeyError(f"Key {key} not found in EncodeOutputDict.")

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.data:
            self.data[key] = value
        else:
            self.additional_data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data or key in self.additional_data

    def __repr__(self):
        return f"DynamicEncodeOutputDict(data={self.data}, additional_data={self.additional_data})"

    def __post_init__(self):
        assert self.ids.ndim == 2, f"ids should be a 2D tensor (batch, token), got: {self.ids.shape}"
        assert self.mask.ndim == 2, f"mask should be a 2D tensor (batch, token), got: {self.mask.shape}"
        if self.position_offsets is not None:
            assert self.position_offsets.ndim == 2, f"position_offsets should be a 2D tensor (batch, token) or None, got: {self.position_offsets.shape}"
    
class GenerateOutputDict(TypedDict):
    decoded_output: List[str]
    output_shape: List
    logits: Optional[torch.Tensor]
    top_logits: Optional[torch.Tensor]

    def __post_init__(self):
        assert (isinstance(decoded_output, list) and isinstance(decoded_output[0], str)), f"decoded_output should be a list of strings, got: {type(decoded_output)}, {type(decoded_output[0])}, content: {decoded_output}"
        if logits != None:
            assert self.logits.ndim == 3, f"logits should be a 3D tensor (batch, token, scores) or None, got: {self.logits.shape}"
            assert self.top_logits.ndim == 4, f"top_logits should be a 4D tensor (batch, token, top_score, index and prob) or None, got: {self.top_logits.shape}"



class BaseHelper(ABC):
    
    @abstractmethod
    def __init__(self, sync, model=None, tokenizer=None, image_processor=None, path_to_model=None):
        pass
        
    @abstractmethod
    def encode(self, inputs: Union[str, List[str]], encode_special_tokens=True) -> EncodeOutputDict:
        pass

    @abstractmethod
    def decode(self, inputs: torch.Tensor, skip_special_tokens=True, logits_separate=False) -> Union[List[str], List[List[str]]]:
        pass
    
    @abstractmethod
    def generate(self, inputs: torch.Tensor, **kwargs) -> GenerateOutputDict:
        pass