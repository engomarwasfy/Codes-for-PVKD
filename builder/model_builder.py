# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py 

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import U2NET, UNET, U2NETP, U2NETP2
from network.cylinder_fea_generator import cylinder_fea


def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']
    cylinder_3d_spconv_seg = None
    if model_config['model_name'] == 'U2NET':
        cylinder_3d_spconv_seg = U2NET(
            output_shape=output_shape,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)
    elif model_config['model_name'] == 'U2NETP':
        cylinder_3d_spconv_seg = U2NETP(
            output_shape=output_shape,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)
    elif model_config['model_name'] == 'U2NETP2':
        cylinder_3d_spconv_seg = U2NETP2(
            output_shape=output_shape,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)
    elif model_config['model_name'] == 'UNET':
        cylinder_3d_spconv_seg = UNET(
            output_shape=output_shape,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)

    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
