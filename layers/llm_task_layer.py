from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3PreTrainedModel, Qwen3RMSNorm

from peft import LoraConfig, get_peft_model, TaskType

class Qwen3ForMatching(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self._tp_plan = {}
        self.config = config
        self.model = Qwen3Model(config)
        self.projections = nn.ModuleDict({})
        self.norms = nn.ModuleDict({})
        self.dropout = nn.Dropout(0.1)

    def init_by_task(self, task_labels, emb_dim=64):
        projections = {}
        norms = {}
        for task, num_labels in task_labels.items():
            if 'CTA' in task:
                projections[task] = nn.Linear(self.config.hidden_size, emb_dim)
                norms[task] = Qwen3RMSNorm(emb_dim, eps=self.config.rms_norm_eps)
            elif 'CPA' in task:
                projections[task] = nn.Linear(self.config.hidden_size * 2, emb_dim)
                norms[task] = Qwen3RMSNorm(emb_dim, eps=self.config.rms_norm_eps)
            else:
                raise ValueError("'CTA' or 'CPA' is not found in task name.")
        self.projections = nn.ModuleDict(projections)
        self.norms = nn.ModuleDict(norms)
        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        task: Optional[str] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        previous_hidden_states = transformer_outputs.last_hidden_state
        if 'CTA' in task:
            hidden_states = transformer_outputs.last_hidden_state
        elif 'CPA' in task:
            hidden_states = transformer_outputs.last_hidden_state
            hidden_states_last_eos = hidden_states[:, -1].unsqueeze(1).repeat([1, hidden_states.shape[1], 1])
            hidden_states = torch.cat([hidden_states_last_eos, hidden_states], 2)
        else:
            raise ValueError("'CTA' or 'CPA' is not found in task name.")
        logits = self.norms[task](self.projections[task](self.dropout(hidden_states)))

        return logits, previous_hidden_states

def list_non_lora_parameters(model, target_modules):

    non_lora_params = []
    for name, param in model.named_parameters():
        if not any(target in name for target in target_modules):
            non_lora_params.append((name, tuple(param.shape)))
    return sorted(non_lora_params)

def test():
    model_name = 'Qwen/Qwen3-Embedding-8B'
    model = Qwen3ForMatching.from_pretrained(
        model_name,
        output_attentions=False,
        output_hidden_states=False,
    )
    # test
    task_labels = {'CTA-DBP': 10, 'CPA-DBP': 10}
    model.init_by_task(task_labels)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # "embed_tokens"
    list_non_lora_parameters(model, target_modules)

    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=target_modules,
        lora_dropout=0.4,
        bias="none",
        init_lora_weights="gaussian",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        print(f"  {name:60} {param.shape}")

    for name, param in model.named_parameters():
        if "poolers" in name or "projections" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

if __name__ == "__main__":
    test()