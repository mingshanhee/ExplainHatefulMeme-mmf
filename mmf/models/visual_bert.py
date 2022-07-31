# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.modules.hf_layers import BertEncoderJit, BertLayerJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.torchscript import getattr_torchscriptable
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import nn, Tensor
from transformers.modeling_bert import (
    BertConfig,
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)


class VisualBERTBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayerJit(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        if (
            self.bypass_transformer
            and visual_embeddings is not None
            and hasattr(self, "additional_layer")
        ):
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output, text_extended_attention_mask
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output[0])
            return final_sequence_output[0], pooled_output, []

        else:
            encoded_layers = self.encoder(embedding_output, extended_attention_mask)
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            attn_data_list: List[Tensor] = []

            if not torch.jit.is_scripting():
                if self.output_attentions:
                    attn_data_list = encoded_layers[1:]
            else:
                assert (
                    not self.output_attentions
                ), "output_attentions not supported in script mode"

            return sequence_output, pooled_output, attn_data_list


class VisualBERTForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = self.config.get("bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.bert.config.vocab_size

        # TODO: Once omegaconf fixes int keys issue, bring this back
        # See https://github.com/omry/omegaconf/issues/149
        # with omegaconf.open_dict(self.config):
        #     # Add bert config such as hidden_state to our main config
        #     self.config.update(self.bert.config.to_dict())
        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                config=self.bert.config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
        self.cls = deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them
        instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment,
        )

        output_dict: Dict[str, Tensor] = {}
        if not torch.jit.is_scripting():
            if self.output_attentions:
                output_dict["attention_weights"] = attention_weights

            if self.output_hidden_states:
                output_dict["sequence_output"] = sequence_output
                output_dict["pooled_output"] = pooled_output
        else:
            assert not (
                self.output_attentions or self.output_hidden_states
            ), "output_attentions or output_hidden_states not supported in script mode"

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        if masked_lm_labels is not None:
            output_dict["logits"] = prediction_scores
            masked_lm_loss = self.loss_fct(
                prediction_scores.contiguous().view(-1, self.vocab_size),
                masked_lm_labels.contiguous().view(-1),
            )
            output_dict["masked_lm_loss"] = masked_lm_loss
            output_dict["loss"] = masked_lm_loss

        return output_dict


class VisualBERTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if self.config.training_head_type == "nlvr2":
            self.bert.config.hidden_size *= 2
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, self.config.num_labels),
        )

        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

        # Set last hidden layer
        if "losses" in self.config and self.config.zerobias:
            for loss in self.config.losses:
                if "bce" in loss["type"]:
                    self.classifier[1].bias.data.fill_(self.config.biasfill)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment,
        )

        if self.training_head_type == "nlvr2":
            # 2B * H => B * 2H
            b, h = pooled_output.size()
            pooled_output = torch.cat(
                [pooled_output[: b // 2], pooled_output[b // 2 :]], dim=1
            )

        output_dict: Dict[str, Tensor] = {}
        if not torch.jit.is_scripting():
            if self.output_attentions:
                output_dict["attention_weights"] = attention_weights

            if self.output_hidden_states:
                output_dict["sequence_output"] = sequence_output
                output_dict["pooled_output"] = pooled_output
        else:
            assert not (
                self.output_attentions or self.output_hidden_states
            ), "output_attentions or output_hidden_states not supported in script mode"

        if self.pooler_strategy == "vqa":
            # In VQA2 pooling strategy, we use representation from second last token
            index_to_gather = input_mask.sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict["scores"] = reshaped_logits
        return output_dict


@registry.register_model("visual_bert")
class VisualBERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.training_head_type: str = self.config.training_head_type

    @classmethod
    def config_path(cls):
        return "configs/models/visual_bert/pretrain.yaml"

    def build(self):
        if self.training_head_type == "pretraining":
            self.model = VisualBERTForPretraining(self.config)
        else:
            self.model = VisualBERTForClassification(self.config)

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def add_post_flatten_params(
        self, image_mask, input_mask
    ) -> Dict[str, Tensor]:
        visual_embeddings_type = torch.zeros_like(image_mask)
        attention_mask = torch.cat(
            (input_mask, image_mask), dim=-1
        )

        return visual_embeddings_type, attention_mask

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, input_ids, token_type_ids, input_mask, image_mask, visual_embeddings, masked_lm_labels) -> Dict[str, Tensor]:
        input_ids = transform_to_batch_sequence(input_ids)
        token_type_ids = transform_to_batch_sequence(token_type_ids)
        input_mask = transform_to_batch_sequence(input_mask)
        image_mask = transform_to_batch_sequence(image_mask)

        visual_embeddings = transform_to_batch_sequence_dim(visual_embeddings)

        return input_ids, token_type_ids, input_mask, image_mask, visual_embeddings, masked_lm_labels

    def update_sample_list_based_on_head(
        self, image_feat_variable
    ) -> Dict[str, Tensor]:
        
        image_dim_variable = image_feat_variable.new_full(
                size=(image_feat_variable.size(0), 1),
                fill_value=image_feat_variable.size(1),
            )

        return image_dim_variable

    def add_custom_params(self, visual_embeddings, image_dim) -> Dict[str, Tensor]:
        masked_lm_labels = None

        # Prepare Mask
        image_mask = torch.arange(
            visual_embeddings.size(-2), device=visual_embeddings.device
        ).expand(visual_embeddings.size()[:-1])
        if len(image_dim.size()) < len(image_mask.size()):
            image_dim = image_dim.unsqueeze(-1)
            assert len(image_dim.size()) == len(image_mask.size())
        image_mask = image_mask < image_dim

        return masked_lm_labels, image_mask.long()

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def forward(self, input_ids, image_feature_0, input_mask, segment_ids) -> Dict[str, Tensor]:

        image_dim = self.update_sample_list_based_on_head(image_feature_0)
        masked_lm_labels, image_mask = self.add_custom_params(image_feature_0, image_dim)
        input_ids, token_type_ids, input_mask, image_mask, visual_embeddings, masked_lm_labels = \
            self.flatten_for_bert(input_ids, segment_ids, input_mask, image_mask, image_feature_0, masked_lm_labels)
        visual_embeddings_type, attention_mask = self.add_post_flatten_params(image_mask, input_mask)

        output_dict = self.model(
            input_ids,
            input_mask,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            None,
            masked_lm_labels
        )

        return output_dict