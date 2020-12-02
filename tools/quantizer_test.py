import os
import argparse
import time
import torch
from nni.compression.torch import apply_compression_results, ModelSpeedup

import anyconfig

torch.manual_seed(0)
use_mask = True
use_speedup = True
compare_results = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser("speedup")
    parser.add_argument("--model_file", type=str, default=None, help="the path of the masks file")
    parser.add_argument('--config_file', default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    args = parser.parse_args()

    import sys
    import pathlib

    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    print(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    print(str(__dir__.parent.parent))
    # project = 'DBNet.pytorch'  # 工作项目根目录
    # sys.path.append(os.getcwd().split(project)[0] + project)

    from utils import parse_config

    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)

    import torch
    from models import build_model
    from data_loader import get_dataloader

    config['distributed'] = False
    config['local_rank'] = 0

    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train']['dataset']['args'][
                                                         'img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])

    # validate_loader = get_dataloader(config['dataset']['validate'], False)
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dummy_input = next(iter(validate_loader))
    # dummy_input = dummy_input['img'].to(device)

    checkpoint = torch.load(args.model_file, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    for key in model.state_dict().keys():
        print(key)
        print(model.state_dict()[key].shape)
        print(model.state_dict()[key])
        break

    import nni

    configure_list = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 8,
        },
        'op_types': ['Conv2d', 'Linear']
    }]

    model = nni.compression.torch.NaiveQuantizer(model, configure_list).compress()

    for key in model.state_dict().keys():
        print(key)
        print(model.state_dict()[key].shape)
        print(model.state_dict()[key])
        break
