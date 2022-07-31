# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from re import L
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
from caffe2.python.timeout_guard import CompleteInTimeOrDie
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import gather_tensor, is_master, is_xla

import torch.nn.functional as F
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from mmf.models.visual_bert import VisualBERT
from mmf.models.vilbert import ViLBERT

import json
import os
import numpy as np
import pickle as pkl


logger = logging.getLogger(__name__)

def process_VisualBERT_attentions(attentions, text_length, vision_length):
    """
        For VisualBERT, there are three attention weights
        - attentions[0]: text-to-text attention weights
        - attentions[1]: vision-to-vision attention weights
        - attentions[2]: text-to-vision and vision-to-text attention weights
    """
    text_attn = []
    vision_attn = []
    text2vision_attn, vision2text_attn = [], []
    for attn in attentions[0]:
        attn = attn.squeeze().clone().detach().cpu().numpy()

        text_attn.append(attn[:text_length, :text_length]) # The first 128 tokens = Text
        vision_attn.append(attn[text_length:, text_length:]) # The subsequent tokens = Image
        text2vision_attn.append(attn[:text_length, text_length:])
        vision2text_attn.append(attn[text_length:, :text_length])

    return {
        "t2t": text_attn,
        "i2i": vision_attn,
        "t2i": text2vision_attn,
        "i2t": vision2text_attn
    }

def process_ViLBERT_attentions(attentions, text_length, vision_length):
    """
        For ViLBERT, there are three attention weights
        - attentions[0]: text-to-text attention weights
        - attentions[1]: vision-to-vision attention weights
        - attentions[2]: text-to-vision and vision-to-text attention weights
    """
    text_attn = []
    for attn in attentions[0]:
        attn = attn.squeeze().clone().detach().cpu().numpy()
        text_attn.append(attn)

    vision_attn = []
    for attn in attentions[1]:
        attn = attn.squeeze().clone().detach().cpu().numpy()
        vision_attn.append(attn)

    text2vision_attn, vision2text_attn = [], []
    for attn_tuple in attentions[2]:
        for attn in attn_tuple:
            attn = attn.squeeze().clone().detach().cpu().numpy()

            if attn.shape[1] == text_length:
                text2vision_attn.append(attn)
            elif attn.shape[1] == vision_length:
                vision2text_attn.append(attn)
            else:
                raise ValueError("Unobserved length...")

    return {
        "t2t": text_attn,
        "i2i": vision_attn,
        "t2i": text2vision_attn,
        "i2t": vision2text_attn
    }


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, dataset_type: str, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        use_cpu = self.config.evaluation.get("use_cpu", False)
        loaded_batches = 0
        skipped_batches = 0

        model_wrapper = ModelInputWrapper(self.model)
        token_reference = TokenReferenceBase(reference_token_idx=102)
        lig = LayerIntegratedGradients(model_wrapper, [model_wrapper.input_maps["image_feature_0"], model_wrapper.module.model.bert.embeddings.word_embeddings])

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            while reporter.next_dataset(flush_report=False):
                dataloader = reporter.get_dataloader()
                combined_report = None

                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader, disable=disable_tqdm)
                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    with CompleteInTimeOrDie(600 if loaded_batches else 3600 * 24):
                        loaded_batches += 1
                        prepared_batch = reporter.prepare_batch(batch)
                        prepared_batch = to_device(prepared_batch, self.device)
                        if not validate_batch_sizes(prepared_batch.get_batch_size()):
                            logger.info("Skip batch due to uneven batch sizes.")
                            skipped_batches += 1
                            continue

                        # Perform normal inference
                        # print(type(self.model), isinstance(self.model, VilBERT))
                        # exit()
                        if isinstance(self.model, VisualBERT):
                            additional_args = (
                                prepared_batch['targets'],
                                prepared_batch['input_mask'], 
                                prepared_batch['segment_ids'], 
                                True
                            )
                        elif isinstance(self.model, ViLBERT):
                            additional_args = (
                                prepared_batch['targets'],
                                prepared_batch['input_mask'], 
                                prepared_batch['segment_ids'], 
                                True,
                                prepared_batch["image_info_0"],
                                prepared_batch["lm_label_ids"]
                            )
                        else:
                            raise NotImplementedError(f"{type(self.model)} not implemented...")

                        scores, model_outputs = self.model(prepared_batch['input_ids'], 
                                                            prepared_batch['image_feature_0'],
                                                            *additional_args)

                        pred, answer_idx = F.softmax(scores, dim=1).data.cpu().max(dim=1)

                        # Prepare references for Layer Integrated Gradients
                        q_reference_indices = token_reference.generate_reference(prepared_batch['input_ids'].shape[1], device='cuda').unsqueeze(0)
                        
                        inputs = (prepared_batch['input_ids'], prepared_batch['image_feature_0'])
                        baselines = (q_reference_indices, prepared_batch['image_feature_0'] * 0.0)


                        if isinstance(self.model, VisualBERT):
                            additional_args = (
                                prepared_batch['targets'],
                                prepared_batch['input_mask'], 
                                prepared_batch['segment_ids'], 
                                False
                            )
                        elif isinstance(self.model, ViLBERT):
                            additional_args = (
                                prepared_batch['targets'],
                                prepared_batch['input_mask'], 
                                prepared_batch['segment_ids'], 
                                False,
                                prepared_batch["image_info_0"],
                                prepared_batch["lm_label_ids"]
                            )
                        else:
                            raise NotImplementedError(f"{type(self.model)} not implemented...")

                        attr, delta = lig.attribute(
                            inputs=inputs,
                            baselines=baselines,
                            n_steps=50,
                            target=1,
                            additional_forward_args=additional_args, 
                            return_convergence_delta=True
                        )

                        metadata = {
                            "pred_probs": pred[0].item(),
                            "pred_class": "hateful" if answer_idx == 1 else "not hateful",
                            "true_class": "hateful" if prepared_batch['targets'].item() == 1 else "not hateful",
                            "raw_input": prepared_batch['text'],
                            "attr_class": "hateful",
                            "convergence_score": delta.item()
                        }
                        
                        attn_dir = self.config["env"]["captum_dir"]

                        filepath = os.path.join(attn_dir, 'metadata', f"{prepared_batch['id'].item()}_text_metadata.json")
                        with open(filepath, 'w') as f:
                            json.dump(metadata, f)

                        filepath = os.path.join(attn_dir, 'gradients', f"{prepared_batch['id'].item()}_text_gradients.npy")
                        with open(filepath, 'wb') as f:
                            np.save(f, attr[1].clone().detach().cpu().numpy())

                        filepath = os.path.join(attn_dir, 'gradients', f"{prepared_batch['id'].item()}_img_gradients.npy")
                        with open(filepath, 'wb') as f:
                            np.save(f, attr[0].clone().detach().cpu().numpy())


                        # Convert tensors to numpy 

                        if isinstance(self.model, VisualBERT):
                            process_func = process_VisualBERT_attentions
                        elif isinstance(self.model, ViLBERT):
                            process_func = process_ViLBERT_attentions
                        else:
                            raise NotImplementedError(f"{type(self.model)} not implemented...")

                        attention_weights = process_func(model_outputs['attention_weights'], prepared_batch['input_ids'].shape[1], prepared_batch['image_feature_0'].shape[1])

                        # temporary code to output attention_weights
                        filepath = os.path.join(attn_dir, 'attentions', f"{prepared_batch['id'].item()}.npy")
                        with open(filepath, 'wb') as handle:
                            pkl.dump(attention_weights, handle, protocol=pkl.HIGHEST_PROTOCOL)

                        model_outputs['scores'] = scores
                        report = Report(prepared_batch, model_outputs)
                        report = report.detach()

                        meter.update_from_report(report)

                        moved_report = report
                        # Move to CPU for metrics calculation later if needed
                        # Explicitly use `non_blocking=False` as this can cause
                        # race conditions in next accumulate
                        if use_cpu:
                            moved_report = report.copy().to("cpu", non_blocking=False)

                        # accumulate necessary params for metric calculation
                        if combined_report is None:
                            # make a copy of report since `reporter.add_to_report` will
                            # change some of the report keys later
                            combined_report = moved_report.copy()
                        else:
                            combined_report.accumulate_tensor_fields_and_loss(
                                moved_report, self.metrics.required_params
                            )
                            combined_report.batch_size += moved_report.batch_size

                        # Each node generates a separate copy of predict JSON from the
                        # report, which will be used to evaluate dataset-level metrics
                        # (such as mAP in object detection or CIDEr in image captioning)
                        # Since `reporter.add_to_report` changes report keys,
                        # (e.g scores) do this after
                        # `combined_report.accumulate_tensor_fields_and_loss`
                        if "__prediction_report__" in self.metrics.required_params:
                            # Still need to use original report here on GPU/TPU since
                            # it will be gathered
                            reporter.add_to_report(report, self.model)

                        if single_batch is True:
                            break

                logger.info(f"Finished training. Loaded {loaded_batches}")
                logger.info(f" -- skipped {skipped_batches} batches.")

                reporter.postprocess_dataset_report()
                assert (
                    combined_report is not None
                ), "Please check if your validation set is empty!"
                # add prediction_report is used for set-level metrics
                combined_report.prediction_report = reporter.report

                combined_report.metrics = self.metrics(combined_report, combined_report)

                # Since update_meter will reduce the metrics over GPUs, we need to
                # move them back to GPU but we will only move metrics and losses
                # which are needed by update_meter to avoid OOM
                # Furthermore, do it in a non_blocking way to avoid any issues
                # in device to host or host to device transfer
                if use_cpu:
                    combined_report = combined_report.to(
                        self.device, fields=["metrics", "losses"], non_blocking=False
                    )

                meter.update_from_report(combined_report, should_update_loss=False)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        skipped_batches = 0
        loaded_batches = 0
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()
                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader)
                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    with CompleteInTimeOrDie(600 if loaded_batches else 3600 * 24):
                        prepared_batch = reporter.prepare_batch(batch)
                        prepared_batch = to_device(prepared_batch, self.device)
                        loaded_batches += 1
                        if not validate_batch_sizes(prepared_batch.get_batch_size()):
                            logger.info("Skip batch due to unequal batch sizes.")
                            skipped_batches += 1
                            continue
                        with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                            model_output = self.model(prepared_batch)
                        report = Report(prepared_batch, model_output)
                        reporter.add_to_report(report, self.model)
                        report.detach()

                reporter.postprocess_dataset_report()

            logger.info(f"Finished predicting. Loaded {loaded_batches}")
            logger.info(f" -- skipped {skipped_batches} batches.")
            self.model.train()

    def _can_use_tqdm(self, dataloader: torch.utils.data.DataLoader):
        """
        Checks whether tqdm can be gracefully used with a dataloader
        1) should have `__len__` property defined
        2) calling len(x) should not throw errors.
        """
        use_tqdm = hasattr(dataloader, "__len__")

        try:
            _ = len(dataloader)
        except (AttributeError, TypeError, NotImplementedError):
            use_tqdm = False
        return use_tqdm


def validate_batch_sizes(my_batch_size: int) -> bool:
    """
    Validates all workers got the same batch size.
    """

    # skip batch size validation on XLA (as there's too much overhead
    # and data loader automatically drops the last batch in XLA mode)
    if is_xla():
        return True

    batch_size_tensor = torch.IntTensor([my_batch_size])
    if torch.cuda.is_available():
        batch_size_tensor = batch_size_tensor.cuda()
    all_batch_sizes = gather_tensor(batch_size_tensor)
    for j, oth_batch_size in enumerate(all_batch_sizes.data):
        if oth_batch_size != my_batch_size:
            logger.error(f"Node {j} batch {oth_batch_size} != {my_batch_size}")
            return False
    return True
