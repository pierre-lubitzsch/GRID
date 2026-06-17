import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import transformers
from torch import nn
from torchmetrics.aggregation import BaseAggregator
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.components.network_blocks.mlp import MLP
from src.models.components.network_blocks.product_key_memory import HashingMemory
from src.models.modules.huggingface.transformer_base_module import TransformerBaseModule
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)


class SemanticIDGenerativeRecommender(TransformerBaseModule):
    """
    This is a base class for the generative recommender model.
    It is used to generate the semantic ID for the given input.
    It does not contain any specific implementation for the encoder or decoder.
    The encoder and decoder are defined in the subclasses.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        embedding_dim: int,
        should_check_prefix: bool,
        top_k_for_generation: int,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDGenerativeRecommender module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        num_embeddings_per_hierarchy (int): the number of embeddings per hierarchy.
        embedding_dim (int): the dimension of the embeddings.
        top_k_for_generation (int): the number of top-k candidates for generation.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """
        super().__init__(**kwargs)

        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks.t()
            assert (
                self.codebooks.size(1) == num_hierarchies
            ), "codebooks should be of shape (-1, num_hierarchies)"
        else:
            logging.warning(
                "Not using pre-cached codebooks, \
            please make sure that \n \
                            1) dataset is properly pre-processed \n \
                            2) num_hierarchies and  num_embeddings_per_hierarchy are proerly set\
            "
            )

        self.top_k_for_generation = top_k_for_generation
        self.forbidden_sids: Optional[Set[Tuple[int, ...]]] = None
        self.filter_mode: str = "global"
        self.user_forbidden_items: Optional[Dict[int, Set[int]]] = None

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject a separator token into the ID embeddings and attention mask.

        Parameters:
        id_embeddings (torch.Tensor): The ID embeddings of shape (batch_size, seq_len, emb_dim).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
        sep_token (torch.Tensor): The separator token of shape (1, emb_dim).
        num_hierarchies (int): The number of hierarchies in the codebooks.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified ID embeddings and attention mask.
        id_embeddings: The ID embeddings with the separator token injected of shape (batch_size, seq_len + num_items, emb_dim).
        attention_mask: The attention mask with the separator token injected of shape (batch_size, seq_len + num_items).

        An intuitive example of the input and output:
        input:
        id_embeddings: [[1, 2, 3, 4], [5, 6, 7, 8]]
        attention_mask: [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        output:
        id_embeddings: [[1, 2, 3, 4, sep_token], [5, 6, 7, 8, sep_token]]
        attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        """
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count_per_sequence = seq_len // num_hierarchies

        reshaped_id_embeddings = id_embeddings.view(
            batch_size, item_count_per_sequence, num_hierarchies, -1
        )
        reshaped_attention_mask = attention_mask.view(
            batch_size, item_count_per_sequence, num_hierarchies
        )
        reshaped_sep_token_for_concat = (
            sep_token.unsqueeze(0)
            .expand(batch_size, item_count_per_sequence, -1)
            .unsqueeze(-2)
        )
        id_embeddings = torch.cat(
            [reshaped_id_embeddings, reshaped_sep_token_for_concat], dim=-2
        )
        attention_mask = torch.cat(
            [reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]],
            dim=-1,
        )
        id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return id_embeddings, attention_mask

    def _spawn_embedding_tables(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        """
        Spawn an embedding table with the given number of embeddings and embedding dimension.

        Parameters:
        num_embeddings (int): the number of embeddings in the table.
        embedding_dim (int): the dimension of the embeddings.
        """
        table = torch.nn.Embedding(
            num_embeddings=num_embeddings,  # type: ignore
            embedding_dim=embedding_dim,  # type: ignore
        )
        return table

    def _is_kv_cache_valid(
        self, kv_cache: Union[Tuple, DynamicCache, EncoderDecoderCache]
    ) -> bool:

        if isinstance(kv_cache, (EncoderDecoderCache, DynamicCache)):
            return len(kv_cache) > 0
        elif isinstance(kv_cache, Tuple):
            return True
        else:
            return False

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Adds repeating offsets to each element in each row of input_sids.
        we use a single embedding table for multiple code books.
        for example if each codebook has 300 embeddings and we have 3 codebooks,
        the input sequence will be transformed from [0, 1, 2] -> to [0, 301, 602]

        Parameters:
            input_sids (torch.Tensor): A 2D PyTorch tensor.
            codebook_size (int): The number of elements in the codebook.
            num_hierarchies (int): The number of hierarchy levels.
        """

        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")

        num_rows, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )

        # Calculate how many times the full offset pattern needs to repeat
        num_repeats = (
            num_cols + num_hierarchies - 1
        ) // num_hierarchies  # Integer division to handle cases where num_cols is not a multiple of num_hierarchies

        # Repeat the offsets and slice to match the number of columns
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

        # Add the repeated offsets to each row using broadcasting
        input_sids_with_offsets = input_sids + repeated_offsets
        if attention_mask is not None:
            input_sids_with_offsets = input_sids_with_offsets * attention_mask
        return input_sids_with_offsets

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """
        Checks if a given prefix is a valid prefix of the codebooks.

        Args:
            prefix: A tensor of shape [batch_size, hierarchy_level].
            batch_size: The size of the batch to process.

        Returns:
            A boolean tensor of shape [batch_size] indicating the validity of each prefix.
        """
        # TODO (clark): this is a temporary solution, we should use a more efficient way to do this
        # like pre-sorting the codebook and implementing a tree strcture

        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        # Ensure codebooks are on the correct device.  Do this *once* outside the loop.
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)

        # Trim the codebooks to the relevant hierarchy *once* outside the loop.
        trimmed_codebooks = self.codebooks[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            # Get the current batch of prefixes.
            batch_prefix = prefix[
                i : i + batch_size
            ]  # Shape: [batch_size, hierarchy_level]

            # Perform the comparison.  Broadcasting is now limited by batch_size.
            # trimmed_codebooks shape: [C, H] -> unsqueezed [C, 1, H]
            # batch_prefix shape   : [b, H] -> unsqueezed [1, b, H]
            # comparison result    : [C, b, H]
            comparison = trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0)

            # Reduce along the hierarchy dimension (H). Shape: [C, b]
            all_match = comparison.all(dim=2)

            # Reduce along the codebook dimension (C).  Shape: [b]
            any_match = all_match.any(dim=0)

            # Append the results for this batch.
            results.append(any_match)

        # Concatenate the results from all batches.
        return torch.cat(results)

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Union[torch.Tensor, None],
        marginal_log_prob: Union[torch.Tensor, None],
        past_key_values: Union[EncoderDecoderCache, None],
        hierarchy: int,
        batch_size: int,
    ):
        """
        Perform one step of beam search.

        Args:
            candidate_logits: The logits for the next token.
            generated_ids: The generated IDs so far.
            marginal_log_prob: The marginal log probabilities.
            past_key_values: The cache for past key values.
            hierarchy: The current hierarchy level.
            batch_size: The size of the batch.

        Returns:
            The updated generated IDs and the marginal probabilities.
        """

        # pruning the beams that cannot be mapped to a valid item
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.num_embeddings_per_hierarchy,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                # we prune all beams with prefixes that cannot be mapped to a valid item
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.num_embeddings_per_hierarchy, dim=0
                            ),
                            torch.arange(
                                self.num_embeddings_per_hierarchy,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.num_embeddings_per_hierarchy)
            candidate_logits[~valid_prefix_mask] = float("-inf")

        # Decode-time filter: mask logits completing a forbidden SID at final hierarchy.
        if (
            self.forbidden_sids
            and generated_ids is not None
            and hierarchy == self.num_hierarchies - 1
        ):
            batch_beams = generated_ids.reshape(-1, hierarchy)
            for beam_idx in range(batch_beams.size(0)):
                prefix = batch_beams[beam_idx].tolist()
                for tok in range(self.num_embeddings_per_hierarchy):
                    sid = tuple(prefix + [tok])
                    if sid in self.forbidden_sids:
                        candidate_logits[beam_idx, tok] = float("-inf")

        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            # we need to overwrite the cache because we expanded the beam width from bsz to bsz * beam_width
            # real KV cache starts from the first hierarchy rather than 0-th
            # this is because in 0th hierarchy, self-attention doesn't have cache.
            # and kv cache in huggingface has poor support for this corner case
            past_key_values = EncoderDecoderCache(
                self_attention_cache=DynamicCache(),
                cross_attention_cache=DynamicCache(),
            )
            replace_indices = None
        else:
            # we have beams, generating more beams from the existing beams
            proba, indices = (
                proba[:, : self.num_embeddings_per_hierarchy],
                indices[:, : self.num_embeddings_per_hierarchy],
            )
            proba, indices = proba.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            ), indices.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            )
            # calculating the marginal probability
            proba = torch.mul(
                marginal_log_prob.repeat_interleave(
                    self.num_embeddings_per_hierarchy, dim=-1
                ),
                proba,
            )
            topk_results = torch.topk(
                torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1
            )
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            # getting indices of winning beams in the original beams
            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()
            # accordingly update kv cache given the winning beams
            if past_key_values != None:
                past_key_values.reorder_cache(replace_indices)

            indices_topk = torch.gather(indices, 1, indices_topk)

        if replace_indices != None:
            generated_ids = torch.cat(
                [
                    generated_ids.reshape(-1, hierarchy)[replace_indices].reshape(
                        -1, self.top_k_for_generation, hierarchy
                    ),
                    indices_topk.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            generated_ids = indices_topk.unsqueeze(-1)

        return generated_ids, proba_topk, past_key_values

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        _, loss = self.model_step(model_input=model_input, label_data=label_data)

        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=list(label_data.labels.values())[0].to(marginal_probs.device),
        )

        loss_to_aggregate(loss)

    def _make_deterministic(self, is_training: bool):
        """
        Make the model deterministic by turning off some flags.
        This is needed as the default functions in lightning such as
        on_validation_start on_predict_start cannnot properly set the flags
        for the encoder and decoder.
        (TODO) clark: in the future we can revisit this and make it more generic

        Args:
            is_training (bool): Whether the model is in training mode or not.
        """
        if is_training:
            if self.decoder != None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
            if self.encoder != None:
                self.encoder.encoder.is_training = True
                self.encoder.encoder.train()
        else:
            if self.decoder != None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()
            if self.encoder != None:
                self.encoder.encoder.is_training = False
                self.encoder.encoder.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)

    def on_train_start(self):
        super().on_train_start()
        self._make_deterministic(is_training=True)


class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        top_k_for_generation: int = 10,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        mlp_layers: Optional[int] = None,
        pkm_layers: Optional[Union[str, Dict[str, Any]]] = None,
        pkm_params: Optional[Dict[str, Any]] = None,
        pkm_mode: str = "replace",
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        adaptive_item_offset_stable_codes: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings_per_hierarchy).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        mlp_layers (Optional[int]): the number of mlp layers in the encoder and decoder.
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """

        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = (
                kwargs["huggingface_model"]
                .encoder.block[0]
                .layer[0]
                .SelfAttention.q.in_features
            )

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        self.decoder = SemanticIDDecoderModule(
            decoder=self.decoder,
            bos_token=bos_token,
            decoder_mlp=torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
        )

        # Replace selected T5 feed-forward sub-layers with Product-Key Memory
        # layers. Runs before the mlp_layers bloating below so that PKM-selected
        # layers stay PKM and only the remaining FFNs get bloated.
        if pkm_layers is not None:
            self._install_pkm_layers(pkm_layers, dict(pkm_params or {}), pkm_mode)

        if mlp_layers is not None:
            # bloating the mlp layers in both encoder and decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.encoder.encoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # separation token for the encoder to differentiate between items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name
        self.repair_adapter: Optional[nn.Parameter] = None
        # Per-item adaptive-position offsets (Stable-Adaptive Semantic IDs,
        # "option 2"). None unless explicitly enabled. When set, a zero-init,
        # per-item, per-adaptive-hierarchy offset is added to the item
        # embeddings at the adaptive positions [stable_codes, num_hierarchies),
        # giving genuinely item-local degrees of freedom while the shared SID
        # table stays frozen. Enabled at construction (for eval/load
        # consistency) via adaptive_item_offset_stable_codes, or lazily during
        # unlearning via enable_adaptive_item_offset().
        self.adaptive_item_offset: Optional[nn.Parameter] = None
        self._adaptive_stable_codes: Optional[int] = None
        if adaptive_item_offset_stable_codes is not None:
            self.enable_adaptive_item_offset(int(adaptive_item_offset_stable_codes))

    def set_decode_filter(
        self,
        *,
        forbidden_sids: Optional[Set[Tuple[int, ...]]] = None,
        filter_mode: str = "global",
        user_forbidden_items: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        self.forbidden_sids = forbidden_sids
        self.filter_mode = filter_mode
        self.user_forbidden_items = user_forbidden_items

    def _install_pkm_layers(
        self,
        pkm_layers: Union[str, Dict[str, Any]],
        pkm_params: Dict[str, Any],
        pkm_mode: str = "replace",
    ) -> None:
        """Add Product-Key Memory to selected T5 feed-forward sub-layers.

        ``pkm_layers`` selects which feed-forward layers are targeted:
            * ``"all"``: every FFN in both encoder and decoder.
            * a mapping ``{"encoder": <sel>, "decoder": <sel>}`` where each
              ``<sel>`` is ``None`` (none), ``"all"``, or a list of 0-indexed
              transformer block ids.
        ``None`` (handled by the caller) means no PKM anywhere.

        ``pkm_mode`` controls how the PKM relates to the existing FFN:
            * ``"replace"``: swap the FFN for a PKM (:class:`T5LayerPKM`).
            * ``"add"``: keep the FFN and run a PKM in parallel on the same
              input, summing their outputs (:class:`T5LayerFFWithPKM`).

        ``pkm_params`` is forwarded to :class:`HashingMemory` (k_dim, heads,
        knn, n_keys, dropouts, query_batchnorm, sparse).
        """
        if pkm_mode not in ("replace", "add"):
            raise ValueError(
                f"Unknown pkm_mode: {pkm_mode!r} (expected 'replace' or 'add')"
            )
        # Collect the FFN module names per subtree, keyed by transformer block id.
        enc_ffns: Dict[int, str] = {}
        dec_ffns: Dict[int, str] = {}
        for name, module in self.named_modules():
            if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                match = re.search(r"block\.(\d+)\.", name)
                if match is None:
                    continue
                block_id = int(match.group(1))
                if name.startswith("encoder"):
                    enc_ffns[block_id] = name
                elif name.startswith("decoder"):
                    dec_ffns[block_id] = name

        enc_ids, dec_ids = _resolve_pkm_selection(
            pkm_layers, sorted(enc_ffns), sorted(dec_ffns)
        )

        config = self.encoder.encoder.config
        target_names = [enc_ffns[i] for i in enc_ids] + [dec_ffns[i] for i in dec_ids]
        for name in target_names:
            parent_module, attr_name = get_parent_module_and_attr(self, name)
            if pkm_mode == "replace":
                new_module = T5LayerPKM(config=config, pkm_params=pkm_params)
            else:  # "add": keep the existing FFN and run a PKM in parallel
                existing_ffn = getattr(parent_module, attr_name)
                new_module = T5LayerFFWithPKM(
                    ffn=existing_ffn, config=config, pkm_params=pkm_params
                )
            setattr(parent_module, attr_name, new_module)

        logging.info(
            "Installed PKM layers (mode=%s) -> encoder blocks %s, decoder blocks %s (params=%s)",
            pkm_mode,
            sorted(enc_ids),
            sorted(dec_ids),
            pkm_params,
        )

    def enable_adaptive_item_offset(self, stable_codes: int) -> None:
        """Create per-item adaptive-position offsets (Stable-Adaptive IDs option 2).

        Adds a zero-initialised, per-item, per-adaptive-hierarchy offset that is
        injected into the item embeddings at the adaptive positions
        ``[stable_codes, num_hierarchies)``. The shared SID table is untouched,
        so these offsets are the only item-local degrees of freedom. Idempotent.

        Also precomputes a vectorised inverse map (SID code tuple -> item id)
        used to apply the offsets in the forward pass.
        """
        if self.adaptive_item_offset is not None:
            return
        if getattr(self, "codebooks", None) is None:
            raise ValueError(
                "enable_adaptive_item_offset requires codebooks (item -> SID map)"
            )
        num_hierarchies = int(self.num_hierarchies)
        codebook_size = int(self.num_embeddings_per_hierarchy)
        stable_codes = int(stable_codes)
        if not 1 <= stable_codes < num_hierarchies:
            raise ValueError(
                f"stable_codes={stable_codes} must satisfy 1 <= stable_codes < "
                f"num_hierarchies={num_hierarchies}"
            )
        num_items = int(self.codebooks.size(0))
        num_adaptive = num_hierarchies - stable_codes
        device = self.item_sid_embedding_table_encoder.weight.device
        self._adaptive_stable_codes = stable_codes
        self.adaptive_item_offset = nn.Parameter(
            torch.zeros(num_items, num_adaptive, self.embedding_dim, device=device)
        )

        # Inverse map: encode each item's code tuple as a unique integer key and
        # keep them sorted for vectorised lookup via searchsorted.
        powers = codebook_size ** torch.arange(num_hierarchies, dtype=torch.long)
        codebooks = self.codebooks.to(torch.long).cpu()  # (num_items, num_hierarchies)
        keys_all = (codebooks * powers).sum(dim=1)  # (num_items,)
        sorted_keys, order = torch.sort(keys_all)
        self.register_buffer("_adaptive_code_powers", powers.to(device), persistent=False)
        self.register_buffer("_adaptive_sorted_keys", sorted_keys.to(device), persistent=False)
        self.register_buffer(
            "_adaptive_sorted_item_ids", order.to(torch.long).to(device), persistent=False
        )

    def _codes_to_item_ids(self, codes_items: torch.Tensor) -> torch.Tensor:
        """Map raw SID code tuples ``(..., num_hierarchies)`` to item ids.

        Returns ``-1`` for tuples that are padded/out-of-range or absent from
        the codebook.
        """
        codebook_size = int(self.num_embeddings_per_hierarchy)
        powers = self._adaptive_code_powers
        invalid = (codes_items < 0).any(dim=-1) | (codes_items >= codebook_size).any(dim=-1)
        keys = (codes_items.clamp(min=0).to(torch.long) * powers).sum(dim=-1)
        sorted_keys = self._adaptive_sorted_keys
        pos = torch.searchsorted(sorted_keys, keys).clamp(max=sorted_keys.numel() - 1)
        match = (sorted_keys[pos] == keys) & (~invalid)
        return torch.where(
            match, self._adaptive_sorted_item_ids[pos], torch.full_like(keys, -1)
        )

    def _inject_adaptive_offsets(
        self, embeds: torch.Tensor, raw_codes: torch.Tensor
    ) -> torch.Tensor:
        """Add per-item offsets at the adaptive positions of each item block.

        No-op unless the offset table is enabled and ``raw_codes`` is aligned to
        full ``num_hierarchies``-token item blocks (so it is skipped during
        incremental generation, where item identity is not yet determined).
        """
        if self.adaptive_item_offset is None:
            return embeds
        num_hierarchies = int(self.num_hierarchies)
        if (
            raw_codes.dim() != 2
            or raw_codes.size(1) == 0
            or raw_codes.size(1) % num_hierarchies != 0
        ):
            return embeds
        stable = int(self._adaptive_stable_codes)
        batch, seq_len = raw_codes.shape
        n_items = seq_len // num_hierarchies
        codes_items = raw_codes.view(batch, n_items, num_hierarchies)
        item_ids = self._codes_to_item_ids(codes_items)  # (batch, n_items)
        valid = (item_ids >= 0).unsqueeze(-1).unsqueeze(-1)
        offs = self.adaptive_item_offset[item_ids.clamp(min=0)]  # (b, n, adaptive, d)
        offs = offs * valid
        emb_items = embeds.view(batch, n_items, num_hierarchies, embeds.size(-1)).clone()
        emb_items[:, :, stable:, :] = emb_items[:, :, stable:, :] + offs
        return emb_items.view(batch, seq_len, embeds.size(-1))

    def _batch_loss_from_model_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
    ) -> torch.Tensor:
        _, loss = self.model_step(*batch)
        return loss

    def _sequence_log_prob(
        self,
        model_input: SequentialModelInputData,
        label_data: SequentialModuleLabelData,
    ) -> torch.Tensor:
        """Approximate ``log p(s_i | h_u)`` as sum of hierarchy log-probs."""
        fut_ids = None
        for label in label_data.labels:
            fut_ids = label_data.labels[label].reshape(model_input.mask.size(0), -1)
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )
        model_output = model_output[:, :-1]
        log_probs = []
        for hierarchy in range(self.num_hierarchies):
            logits = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            log_p = torch.nn.functional.log_softmax(logits, dim=-1)
            target = fut_ids[:, hierarchy].long()
            log_probs.append(log_p.gather(1, target.unsqueeze(1)).squeeze(1))
        return torch.stack(log_probs, dim=1).sum(dim=1).mean()

    def compute_uniform_kl_loss(
        self,
        model_input: SequentialModelInputData,
        label_data: SequentialModuleLabelData,
    ) -> torch.Tensor:
        """Symmetric-KL loss pushing the model's next-token distribution toward
        uniform on the given (forget) batch.

        TIGER adaptation of ERASE's
        ``Trainer.unlearn_iterative_uniform_distribution``: where the reference
        drives the *item* softmax toward uniform via
        ``KLDiv(log p_model, uniform)``, TIGER is a token-generative model, so
        we drive each hierarchy's next-token softmax (over the codebook
        vocabulary of size ``num_embeddings_per_hierarchy``) toward uniform and
        average across hierarchies. This is the Fanchuan stage-1 objective.
        """
        fut_ids = None
        for label in label_data.labels:
            fut_ids = label_data.labels[label].reshape(model_input.mask.size(0), -1)
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )
        # drop the trailing position that pairs with the prepended bos token
        model_output = model_output[:, :-1]
        kl = torch.nn.KLDivLoss(reduction="batchmean")
        loss = model_output.new_zeros(())
        for hierarchy in range(self.num_hierarchies):
            logits = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            probs = torch.nn.functional.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / probs.size(-1)
            # KLDivLoss expects log-probabilities as the first argument and
            # plain probabilities as the target (ERASE ``kl_loss_sym``).
            loss = loss + kl(torch.log(probs + 1e-20), uniform)
        return loss / self.num_hierarchies

    def _pooled_user_representation(
        self,
        model_input: SequentialModelInputData,
    ) -> torch.Tensor:
        user_id = model_input.transformed_sequences.get("user_id")
        # transformed_sequences uses original feature names ("sequence_data"), not
        # mapped names ("input_ids"); try the raw key first, then the mapped fallback.
        input_ids = model_input.transformed_sequences.get("sequence_data")
        if input_ids is None:
            input_ids = model_input.transformed_sequences.get(
                self.feature_to_model_input_map.get("sequence_data", "input_ids")
            )
        enc_out, enc_mask = self.encoder_forward_pass(
            attention_mask=model_input.mask,
            input_ids=input_ids,
            user_id=user_id,
        )
        mask = enc_mask.unsqueeze(-1).float()
        pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def _item_sid_embedding(self, item_sid_rows: torch.Tensor) -> torch.Tensor:
        """Aggregate encoder SID embeddings for item semantic-id rows."""
        table = self.get_embedding_table(table_name="encoder")
        shifted = self._add_repeating_offset_to_rows(
            input_sids=item_sid_rows,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
        )
        emb = table(shifted)
        return emb.mean(dim=1)

    def _item_encoder_representation(self, item_sid_rows: torch.Tensor) -> torch.Tensor:
        """Pooled encoder output for each item, in the SAME representation space
        as :meth:`_pooled_user_representation` (``r_u``).

        Each item's semantic-id row is encoded as a length-1 history (no user
        token), and the encoder output is masked-mean-pooled — exactly the
        pooling used for ``r_u`` — so item and user representations are
        comparable for the ``L_sep`` contrastive similarity.
        """
        device = next(self.parameters()).device
        # SID rows must be integer indices (labels may arrive as float), and the
        # attention mask must be integer too: encoder_forward_pass multiplies the
        # SID indices by the mask, so a float mask would corrupt the dtype.
        item_sid_rows = item_sid_rows.to(device).long()
        if item_sid_rows.dim() == 1:
            item_sid_rows = item_sid_rows.unsqueeze(0)
        attn = torch.ones(
            item_sid_rows.size(0), item_sid_rows.size(1), dtype=torch.long, device=device
        )
        enc_out, enc_mask = self.encoder_forward_pass(
            attention_mask=attn, input_ids=item_sid_rows, user_id=None
        )
        mask = enc_mask.unsqueeze(-1).float()
        return (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def compute_sep_loss(
        self,
        retain_batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        *,
        negative_item_ids: Set[int],
        temperature: float = 0.007,
    ) -> torch.Tensor:
        """Separation loss ``L_sep'`` (per-user, all-positives form).

        Samples over retain users (one per batch row, visited once per epoch),
        pools the per-item encoder representations of every item in a user's
        history into a single user vector ``r_u``, and then pushes ``r_u``
        toward each of that user's own (positive) history items ``i⁺`` and away
        from the forget items ``I_f``:

            L_sep' = mean_{i⁺ ∈ history(u)}
                -log( exp(sim(r_u, z_i⁺)/τ)
                      / (exp(sim(r_u, z_i⁺)/τ)
                         + Σ_{i_f ∈ I_f} exp(sim(r_u, z_i_f)/τ)) )

        ``r_u`` is the average pooling of the per-item encoder representations of
        the user's history items, and the positives ``z_i⁺`` and the forget
        negatives ``z_i_f`` live in that SAME pooled-encoder space
        (:meth:`_item_encoder_representation`). Negatives are exactly
        ``negative_item_ids`` (the forget set ``I_f`` by default) — no neighbors
        are added. The loss averages over a user's positive items and then over
        the users in the batch.
        """
        model_input, _ = retain_batch
        device = next(self.parameters()).device

        neg_ids = list(negative_item_ids)
        if not neg_ids or self.codebooks is None:
            return torch.zeros((), device=device)

        # --- history item SID rows: [B, n_items, H] (raw codes) ---
        # transformed_sequences uses the original feature name ("sequence_data"),
        # not the mapped name ("input_ids"); try the raw key first.
        input_ids = model_input.transformed_sequences.get("sequence_data")
        if input_ids is None:
            input_ids = model_input.transformed_sequences.get(
                self.feature_to_model_input_map.get("sequence_data", "input_ids")
            )
        input_ids = input_ids.to(device).long()
        mask = model_input.mask.to(device)

        n_hier = int(self.num_hierarchies)
        bsz, seq_len = input_ids.shape
        if seq_len % n_hier != 0:
            raise ValueError(
                f"sequence length {seq_len} not divisible by num_hierarchies "
                f"{n_hier}; cannot split history into items for L_sep"
            )
        n_items = seq_len // n_hier
        item_sids = input_ids.view(bsz, n_items, n_hier)
        # An item is a valid (non-padded) positive iff all its SID tokens attend.
        item_valid = (
            (mask.view(bsz, n_items, n_hier) > 0).sum(dim=-1) == n_hier
        ).float()  # [B, n_items]

        # --- per-item encoder representations z_i: [B, n_items, d] ---
        z_items = torch.nn.functional.normalize(
            self._item_encoder_representation(item_sids.reshape(bsz * n_items, n_hier)),
            dim=-1,
        ).view(bsz, n_items, -1)

        # --- r_u: average pooling of the user's item representations: [B, d] ---
        n_valid = item_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        r_u = (z_items * item_valid.unsqueeze(-1)).sum(dim=1) / n_valid
        r_u = torch.nn.functional.normalize(r_u, dim=-1)  # [B, d]

        # --- forget negatives z_f: [N_neg, d] ---
        neg_sids = self.codebooks[torch.tensor(neg_ids)].to(device)
        z_neg = torch.nn.functional.normalize(
            self._item_encoder_representation(neg_sids), dim=-1
        )  # [N_neg, d]

        tau = float(temperature)
        # sim(r_u, z_i⁺) for every (user, history-item) pair, and sim against I_f.
        pos_sim = (r_u.unsqueeze(1) * z_items).sum(dim=-1) / tau   # [B, n_items]
        neg_logits = torch.mm(r_u, z_neg.t()) / tau               # [B, N_neg]
        neg_lse = torch.logsumexp(neg_logits, dim=-1)             # [B]
        # per-positive InfoNCE: -log( e^{pos} / (e^{pos} + Σ_f e^{neg}) )
        denom_lse = torch.logaddexp(pos_sim, neg_lse.unsqueeze(1))  # [B, n_items]
        per_pos = denom_lse - pos_sim                              # [B, n_items]

        # Average over a user's valid positives, then over users with ≥1 positive.
        per_user = (per_pos * item_valid).sum(dim=1) / item_valid.sum(dim=1).clamp(
            min=1.0
        )
        has_pos = (item_valid.sum(dim=1) > 0).float()
        return (per_user * has_pos).sum() / has_pos.sum().clamp(min=1.0)

    def compute_unified_loss(
        self,
        *,
        retain_batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        forget_batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        lambda_forget: float = 1.0,
        lambda_sep: float = 0.1,
        forget_loss_level: str = "token",
        sep_temperature: float = 0.007,
        deletion_spec: str = "session",
        forget_item_ids: Optional[Set[int]] = None,
        neighbor_item_ids: Optional[Set[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        l_retain = self._batch_loss_from_model_step(retain_batch)
        if str(forget_loss_level).lower() == "sequence":
            # Minimize log p(s_i|h_u) on forget set.
            l_forget = self._sequence_log_prob(*forget_batch)
        else:
            # CE = -log p; negate so minimizing total suppresses forget targets.
            l_forget = -self._batch_loss_from_model_step(forget_batch)
        l_sep = self.compute_sep_loss(
            retain_batch,
            negative_item_ids=forget_item_ids or set(),
            temperature=sep_temperature,
        )
        total = l_retain + float(lambda_forget) * l_forget + float(lambda_sep) * l_sep
        return {
            "total": total,
            "retain": l_retain,
            "forget": l_forget,
            "sep": l_sep,
        }

    def compute_neighbor_suppression_loss(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        neighbor_item_ids: Set[int],
    ) -> torch.Tensor:
        if not neighbor_item_ids or self.codebooks is None:
            return torch.zeros((), device=next(self.parameters()).device)
        model_input, _ = batch
        r_u = self._pooled_user_representation(model_input)
        losses = []
        for nid in neighbor_item_ids:
            if nid < 0 or nid >= self.codebooks.size(0):
                continue
            sid = self.codebooks[nid].unsqueeze(0)
            z = self._item_sid_embedding(sid)
            losses.append(torch.nn.functional.cosine_similarity(r_u, z.expand_as(r_u)))
        if not losses:
            return torch.zeros((), device=r_u.device)
        return torch.stack(losses).mean()

    def compute_neighborhood_mass_loss(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        neighbor_item_ids: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        if not neighbor_item_ids:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.compute_neighbor_suppression_loss(batch, neighbor_item_ids)

    def compute_prefix_repair_loss(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        neighbor_item_ids: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        return self.compute_neighbor_suppression_loss(
            batch, neighbor_item_ids or set()
        )

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.get_embedding_table(table_name="encoder")(
            shifted_sids
        )
        # Per-item adaptive offsets (option 2); no-op unless enabled. Applied
        # before sep-token injection so item blocks are still aligned to raw_codes.
        inputs_embeds_for_encoder = self._inject_adaptive_offsets(
            inputs_embeds_for_encoder, input_ids
        )

        if self.sep_token is not None:
            (
                inputs_embeds_for_encoder,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds_for_encoder,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        # we enter this loop if we want to use user_id
        if user_id is not None and self.user_embedding is not None:
            # preprocessing function pad user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # prepending the user_id embedding to the input senquence
            inputs_embeds_for_encoder = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds_for_encoder,
                ],
                dim=1,
            )
            # prepending 1 to attention mask as we introduce user embedding in the first column
            user_attention_mask = torch.ones(
                attention_mask.size(0), 1, device=attention_mask.device
            )
            attention_mask_for_encoder = torch.cat(
                [
                    user_attention_mask,
                    attention_mask,
                ],
                dim=1,
            )
        else:
            attention_mask_for_encoder = attention_mask

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
        )
        return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        future_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask_for_encoder: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            encoder_output (Optional[torch.Tensor]): The output from the encoder.
            attention_mask_for_encoder (Optional[torch.Tensor]): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (Optional[DynamicCache]): The cache for past key values.
        """

        # we generated something before and we need to shift the future_ids
        if future_ids is not None:
            shifted_future_sids = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids, device=future_ids.device)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds_for_decoder = self.get_embedding_table(table_name="decoder")(
                shifted_future_sids
            )
            # Per-item adaptive offsets (option 2); no-op unless enabled. Skipped
            # during incremental generation (future_ids not a full item block).
            inputs_embeds_for_decoder = self._inject_adaptive_offsets(
                inputs_embeds_for_decoder, future_ids
            )

            # we do not have valid kv cache
            # we need to prepend bos token to the decoder input
            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds_for_decoder = torch.cat(
                    [
                        self.decoder.bos_token.unsqueeze(0).expand(
                            future_ids.size(0), 1, -1
                        ),
                        inputs_embeds_for_decoder,
                    ],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                # we have valid kv cache
                # we only need the last token in the decoder input
                inputs_embeds_for_decoder = inputs_embeds_for_decoder[:, -1:, :]
        # this is the beginning of generation, we start from bos token
        else:
            inputs_embeds_for_decoder = self.decoder.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        decoder_output = self.decoder(
            sequence_embedding=inputs_embeds_for_decoder,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask_for_encoder,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return decoder_output

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )

        # initilize cached generated ids to None
        generated_ids = None
        marginal_log_prob = None

        # initialize kv cache
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(), cross_attention_cache=DynamicCache()
        )

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                # we generated something before
                # we need to reshape the generated ids so that
                # the number of beams equals to batch size * top_k
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                    encoder_output.device
                )  # shape: (batch_size * top_k, hierarchy)

                repeated_encoder_output = encoder_output.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                # shape: (batch_size * top_k, seq_len+1, hidden_dim)
                # +1 because we have user_id token

                repeated_encoder_attention_mask = (
                    encoder_attention_mask.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
                )  # shape: (batch_size * top_k, seq_len+1)
            else:
                # we haven't generated anything yet!
                # the number of beams currently equals to batch size
                squeezed_generated_ids = None
                repeated_encoder_output = encoder_output
                repeated_encoder_attention_mask = encoder_attention_mask

            # feeding the decoder with the generated ids
            decoder_output, past_key_values = self.decoder_forward_pass(
                future_ids=squeezed_generated_ids,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]

            # # calculating the logits for the next token
            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                latest_output_representation
            )  # shape: (batch_size * top_k, num_embeddings in the hierarchy)

            (
                generated_ids,
                marginal_log_prob,
                past_key_values,
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )

        return generated_ids, marginal_log_prob

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        attention_mask_decoder: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.
        Parameters:
            attention_mask_encoder (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            attention_mask_decoder (Optional[torch.Tensor]): The attention mask for the decoder.
        """

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
            user_id=user_id,
        )

        decoder_output = self.decoder_forward_pass(
            future_ids=future_ids,
            attention_mask=attention_mask_decoder,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,  # we are not using cache for training
        )
        return decoder_output

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "encoder":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "decoder":
            embedding_table = self.item_sid_embedding_table_encoder

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        generated_sids, _ = self.model_step(batch)
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_sids,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the model and calculate the loss if label_data is provided.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.
        """

        # if label_data is None, we are in inference mode and doing free-form generation
        if label_data is None:
            # this is inference stage
            generated_ids, marginal_probs = self.generate(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        # here we pass labels in to the forward function
        # because the decoder is causal and we are doing shifted prediction
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        # we prepended a bos token to the decoder input
        # so we need to remove the last token in the output
        model_output = model_output[:, :-1]

        # the label locations is shared for all semantic id hierarchies
        loss = 0
        for hierarchy in range(self.num_hierarchies):

            input = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            loss += self.loss_function(
                input=input,
                target=fut_ids[:, hierarchy].long(),
            )
        return model_output, loss


class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        decoder_mlp: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        decoder_mlp (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
        bos_token (Optional[torch.nn.Parameter]):
            the bos token used to prompt the decoder.
            if None, then this means the decoder is used standalone without an encoder.
        """

        super().__init__()
        # some sanity checks
        if bos_token is not None:
            assert decoder.config.is_decoder == True, "Decoder must be a decoder model"
            assert (
                decoder.config.is_encoder_decoder == False
            ), "Decoder must be a standalone decoder model"

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.decoder_mlp = decoder_mlp
        # deleting embedding table in the decoder to save space
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        reset_parameters(self.decoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            sequence_embedding (torch.Tensor): The input sequence embedding for the decoder.
            encoder_output (torch.Tensor): The output from the encoder.
            encoder_attention_mask (torch.Tensor): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache): The cache for past key values.
        """

        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=sequence_embedding,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            return embeddings, decoder_outputs.past_key_values
        return embeddings


class SemanticIDEncoderModule(torch.nn.Module):
    """
    This is an in-house replication of the encoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
    ) -> None:
        """
        Initialize the SemanticIDEncoderModule module.

        Paremeters:
        encoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        """
        super().__init__()

        self.encoder = encoder
        embedding_table_dim = find_module_shape(self.encoder, "embed_tokens")
        num_embeddings, embedding_dim = embedding_table_dim

        self.num_embeddings_per_hierarchy = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO (clark): take care of chunky position encoding

        # deleting embedding table in the encoder to save space
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
        reset_parameters(self.encoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
    ) -> torch.Tensor:

        encoder_output = self.encoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
        )
        embeddings = encoder_output.last_hidden_state
        return embeddings


def _resolve_pkm_selection(
    pkm_layers: Union[str, Dict[str, Any]],
    enc_ids: List[int],
    dec_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """Resolve a ``pkm_layers`` selector into concrete (encoder, decoder) block ids.

    ``enc_ids`` / ``dec_ids`` are the sorted lists of available FFN block ids in
    the encoder and decoder, respectively.
    """

    def resolve(sel: Any, available: List[int]) -> List[int]:
        if sel is None:
            return []
        if isinstance(sel, str):
            if sel.lower() == "all":
                return list(available)
            raise ValueError(f"Unknown pkm layer selector: {sel!r} (expected 'all', None, or a list)")
        return [int(x) for x in sel]

    if isinstance(pkm_layers, str):
        if pkm_layers.lower() == "all":
            return list(enc_ids), list(dec_ids)
        raise ValueError(f"Unknown pkm_layers selector: {pkm_layers!r} (expected 'all', None, or a mapping)")

    # mapping form (dict / OmegaConf DictConfig): {"encoder": <sel>, "decoder": <sel>}
    return (
        resolve(pkm_layers.get("encoder"), enc_ids),
        resolve(pkm_layers.get("decoder"), dec_ids),
    )


class T5LayerPKM(nn.Module):
    """A T5 feed-forward sub-layer whose MLP is replaced by a Product-Key Memory.

    Mirrors the contract of ``transformers`` ``T5LayerFF``: it applies a layer
    norm, runs the (memory) transformation, and adds the result back to the
    input as a residual. This makes it a drop-in replacement for ``T5LayerFF``.
    """

    def __init__(self, config: T5Config, pkm_params: Dict[str, Any]) -> None:
        super().__init__()
        self.memory = HashingMemory(
            input_dim=config.d_model,
            output_dim=config.d_model,
            **pkm_params,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.memory(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerFFWithPKM(nn.Module):
    """A T5 feed-forward sub-layer running an FFN and a PKM in parallel.

    The original feed-forward module and the Product-Key Memory both read the
    same input ``hidden_states`` and their contributions are summed:

        out = FFN_layer(h) + dropout(PKM(layer_norm(h)))

    where ``FFN_layer(h) = h + dropout(FFN(ln(h)))`` is the unchanged T5
    feed-forward sub-layer (so there is a single residual on ``h``). This is the
    ``"add"`` counterpart to :class:`T5LayerPKM` (``"replace"``).
    """

    def __init__(
        self, ffn: nn.Module, config: T5Config, pkm_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.ffn = ffn
        self.memory = HashingMemory(
            input_dim=config.d_model,
            output_dim=config.d_model,
            **pkm_params,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ffn_out = self.ffn(hidden_states)
        memory_out = self.memory(self.layer_norm(hidden_states))
        return ffn_out + self.dropout(memory_out)


# TODO (clark): this is a T5 specific implementation
# this class is used for bloating the mlp layers in the encoder and decoder
# original T5 implementation only has one layer
class T5MultiLayerFF(nn.Module):
    def __init__(self, config: T5Config, num_layers: int):
        """
        Initialize the T5MultiLayerFF module.
        This module is a multi-layer feed-forward network (MLP) used in the T5 model.
        It consists of a series of linear layers with ReLU activation and dropout.
        And it also includes layer normalization and residual connections.
        Parameters:
            config (T5Config): The T5 configuration object.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim_list=[config.d_ff for _ in range(num_layers)],
            activation=nn.ReLU,
            dropout=config.dropout_rate,
        )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T5MultiLayerFF module.
        Parameters:
            hidden_states (torch.Tensor): The input hidden states for the MLP.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
