from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model.bert_classifier_with_precalc_representations_agg_chunks import \
    ClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks


def get_model(config, user_info, item_info, device=None, dataset_config=None):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=False, use_user_bias=False)
    elif config['name'] == "MF_with_itembias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=True, use_user_bias=False)
    elif config['name'] == "MF_with_userbias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=False, use_user_bias=True)
    elif config['name'] == "MF_with_itembias_userbias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=True, use_user_bias=True)
    elif config['name'] == "VanillaBERT_precalc_embed_sim":
        model = ClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                               users=user_info,
                                                                               items=item_info,
                                                                               device=device,
                                                                               dataset_config=dataset_config,
                                                                               use_ffn=False,
                                                                               use_item_bias=False,
                                                                               use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_ffn":
        model = ClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                               users=user_info,
                                                                               items=item_info,
                                                                               device=device,
                                                                               dataset_config=dataset_config,
                                                                               use_ffn=True,
                                                                               use_item_bias=False,
                                                                               use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_itembias":
        model = ClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=False,
                                                                                      use_item_bias=True,
                                                                                      use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_ffn_itembias":
        model = ClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=True,
                                                                                      use_item_bias=True,
                                                                                      use_user_bias=False)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
