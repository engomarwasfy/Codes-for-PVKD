# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py
from network.generic_coordinate_mapper_and_sparse_segmentor import get_model_class
from network.torchsparse_segmentor import Asymm_3d_torchsparse
from network.cylinder_fea_generator import cylinder_fea


def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']

    cylinder_3d_spconv_seg = Asymm_3d_torchsparse(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)

    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    model = get_model_class(model_config["model_architecture"])(
        coordinate_transformer=cy_fea_net,
        segmentor= cylinder_3d_spconv_seg,
        sparse_world_shape=output_shape
    )

    return model
 #  segmentator_spconv=cylinder_3d_spconv_seg,
      # segmentator_spconv= u2net.U2NET_full(),