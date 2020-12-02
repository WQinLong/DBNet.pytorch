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
    parser.add_argument("--example_name", type=str, default="slim", help="the name of pruning example")
    parser.add_argument("--masks_file", type=str, default=None, help="the path of the masks file")
    parser.add_argument("--model_file", type=str, default=None, help="the path of the masks file")
    parser.add_argument('--config_file', default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
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
    config['local_rank'] = args.local_rank

    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train']['dataset']['args'][
                                                         'img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])

    validate_loader = get_dataloader(config['dataset']['validate'], False)
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dummy_input = next(iter(validate_loader))
    dummy_input = dummy_input['img'].to(device)

    checkpoint = torch.load(args.model_file, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    use_mask_out = use_speedup_out = None
    # must run use_mask before use_speedup because use_speedup modify the model
    if use_mask:
        apply_compression_results(model, args.masks_file, device)
        start = time.time()
        for _ in range(32):
            use_mask_out = model(dummy_input)
        print('elapsed time when use mask: ', time.time() - start)
    if use_speedup:
        m_speedup = ModelSpeedup(model, dummy_input, args.masks_file, device)
        m_speedup.speedup_model()
        start = time.time()
        for _ in range(32):
            use_speedup_out = model(dummy_input)
        print('elapsed time when use speedup: ', time.time() - start)
    torch.save(model.state_dict(), "output/DBNet_opensource_nni_resnet18_fpn_db/checkpoint/pruner_speed.pth")
    if compare_results:
        if torch.allclose(use_mask_out, use_speedup_out, atol=1e-07):
            print('the outputs from use_mask and use_speedup are the same')
        else:
            raise RuntimeError('the outputs from use_mask and use_speedup are different')
