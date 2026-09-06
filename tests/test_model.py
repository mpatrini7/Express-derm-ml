from express_derm_ml.model import create_model


def test_efficientnet_b2_has_one_logit_output() -> None:
    model = create_model("efficientnet_b2", pretrained=False)
    assert model.classifier[-1].out_features == 1
