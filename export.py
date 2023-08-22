import torch
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def MSMHDA_onnx_export(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
):
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
    sampling_offsets = self.sampling_offsets(query).view(
        N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
    )
    attention_weights = self.attention_weights(query).view(
        N, Len_q, self.n_heads, self.n_levels * self.n_points
    )
    attention_weights = F.softmax(attention_weights, -1).view(
        N, Len_q, self.n_heads, self.n_levels, self.n_points
    )
    # N, Len_q, n_heads, n_levels, n_points, 2
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / self.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    else:
        raise ValueError(
            "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                reference_points.shape[-1]
            )
        )

    output = multi_scale_deformable_attn_pytorch(
        value, input_spatial_shapes, sampling_locations, attention_weights
    )
    output = self.output_proj(output)
    return output
