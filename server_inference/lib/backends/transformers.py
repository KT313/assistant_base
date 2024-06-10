from .base import *


class TransformersHelper(BaseHelper):

    def __init__(self, sync, model, tokenizer, image_processor=None):
        self.sync = sync
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def encode(self, inputs: Union[str, List[str]], encode_special_tokens=True) -> EncodeOutputDict:
        encoded_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.sync.config['torch_device'])
        output = EncodeOutputDict(
            ids = encoded_ids,
            mask = torch.ones_like(encoded_ids, device=self.sync.config['torch_device'])
        )
        return output
        

    def decode(self, inputs: torch.Tensor, skip_special_tokens=True, logits_mode=False) -> Union[List[str], List[List[str]]]:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        print("inputs:", inputs)
        if logits_mode:
            decoded = [self.tokenizer.batch_decode(entry, skip_special_tokens=skip_special_tokens) for entry in inputs]
        else:
            decoded = [self.tokenizer.batch_decode(inputs, skip_special_tokens=skip_special_tokens)]
        if logits_mode:
            decoded = decoded[0]
        print("decoded:", decoded)
        return decoded
        
    
    def generate(self, inputs: torch.Tensor, **kwargs) -> GenerateOutputDict:
        out = self.model.generate(inputs, **kwargs)

        input_sequence_length = inputs.shape[-1]
        decoded = self.decode(out.sequences[:, input_sequence_length:], skip_special_tokens=True)
        output_shape = out.sequences.shape
        merker_scores = []
        if out.scores != None:
            for i in range(len(out.scores)):
                merker_scores.append(out.scores[i])
            scores = torch.stack(merker_scores)
            scores = scores.permute(1, 0, 2) # so the order is batch, token, scores
            print("scores:", scores.shape)
            print(scores)
            # logits = torch.softmax(scores, dim=-1)
            # print("logits:", logits.shape)
            # print(logits)
            top_logits = find_top_indexes(self.sync, scores)
            print("top_logits:", top_logits.shape)
            print("top_logits content:")
            print(top_logits)
            print("sums:")
            for i in range(top_logits.shape[0]):
                print(torch.sum(top_logits[i, 0, :, 1]))
        else:
            logits = None
            top_logits = None

        
        output = GenerateOutputDict(
            decoded_output = decoded,
            output_shape = output_shape,
            logits = None,
            top_logits = top_logits
        )
        for key, val in output.items():
            try:
                print(key, val.shape)
            except:
                print(key, "-")
        
        return output