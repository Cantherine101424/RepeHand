import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from main.config import cfg
from common.base import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids',default='3')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def test(test_epoch):
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    tester = Tester(test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        # save output
        out = {k: v.cpu().numpy() for k, v in out.items()}
        for k, v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += len(out)

    out_eval_result = tester._print_eval_result(eval_result)

    return out_eval_result


def safe_nanmean(data):
    valid_data = []
    for x in data:
        if x is not None:
            try:
                if not np.isnan(x):
                    valid_data.append(x)
            except:
                continue
    return np.mean(valid_data) if valid_data else 0.0


def main():
    eval_result = test(79) # test epoch

if __name__ == "__main__":
    main()
