from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder
dependencies = ["minerva", "torch"]


def harsccencoder(pretrained=False):
    print(pretrained)
    return HARSCnnEncoder(dim=125, input_channel=6, inner_conv_output_dim=128*10)