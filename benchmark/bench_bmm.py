import torch
from torch_int._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_s32t, bmm_s8t_s8n_f32t
from torch_int._CUDA import bmm_s8t_s8n_s32t_cublas
from torch_int._CUDA import bmm_s4t_s4n_f32t
from utils import bench_func_latency
import argparse
import faulthandler
from icecream import ic 



def bench_bmm(precision, batch_size, seq_len, hidden_dim, fn=bmm_s8t_s8n_s32t_cublas):
    
    
    if precision == 'int8':
        a = torch.randint(-128, 127, (batch_size, seq_len,
                          hidden_dim), dtype=torch.int8).cuda()
        b = torch.randint(-128, 127, (batch_size, seq_len,
                          hidden_dim), dtype=torch.int8).cuda()
        scale = 0.01
        args = (a, b)
        # fn = bmm_s8t_s8n_s32t
    elif precision == 'fp16':
        a = torch.randn(batch_size, seq_len, hidden_dim).half().cuda()
        b = torch.randn(batch_size, seq_len, hidden_dim).half().cuda().transpose(1, 2)
        args = (a, b)
        fn = torch.bmm
    elif precision == 'fp32':
        a = torch.randint(6, 8, (batch_size, seq_len,
                          hidden_dim), dtype=torch.int).float()
        a = a.cuda().contiguous()
        b = torch.randint(6, 8, (batch_size, seq_len,
                          hidden_dim), dtype=torch.int).float().cuda().contiguous()
        
        print(b.size())

        c = bmm_s4t_s4n_f32t(a, b, 1.0)
        c_gt0= torch.bmm(a,b.transpose(1,2))
        c_gt = torch.bmm(a,b.transpose(1,2))
        
        # print(c_gt0[0])
        print(c[0])
        print(c_gt[0])
        
        ic(torch.allclose(c, c_gt))
        
        return 
    else:
        raise NotImplementedError
    
    ms = bench_func_latency(fn, args, num_iter=5000)
    faulthandler.enable()

    workload = batch_size * seq_len * seq_len * hidden_dim * 2 * 1.0
    gops = (workload / 1e9) / (ms / 1e3)
    print(f"gops = {gops}")


def test_case(batchSize, seq_len, hidden_dim, precision, fn):

    print(f'B={batchSize}, L={seq_len}, H={hidden_dim}, precision={precision}, func={fn.__name__}')
    bench_bmm(precision, batchSize, seq_len, hidden_dim, fn)
    print("------------------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=12288)
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()
    # print(f'B={args.batch_size}, L={args.seq_len}, H={args.hidden_dim}, precision={args.precision}')
    # bench_bmm(args.precision, args.batch_size, args.seq_len, args.hidden_dim)
    # bench_bmm(args.precision, args.batch_size, args.seq_len, args.hidden_dim, bmm_s8t_s8n_s32t)
    
    # test_case(32, 1024, 512, 'fp32', bmm_s4t_s4n_f32t)
    test_case(32, 256, 64, 'fp32', bmm_s4t_s4n_f32t)

    # test_case(32, 256, 64, 'fp32', bmm_s4t_s4n_f32t)
    
    # test_case(32, 2048, 64, 'int8', bmm_s8t_s8n_s32t)
    # test_case(32, 2048, 64, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(32, 2048, 1024, 'int8', bmm_s8t_s8n_s32t)
    # test_case(32, 2048, 1024, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(1, 512, 12288, 'int8', bmm_s8t_s8n_s32t)
    # test_case(1, 512, 12288, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(1, 2048, 64, 'int8', bmm_s8t_s8n_s32t)
    # test_case(1, 2048, 64, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(1, 2048, 1024, 'int8', bmm_s8t_s8n_s32t)
    # test_case(1, 2048, 1024, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(8, 512, 12288, 'int8', bmm_s8t_s8n_s32t)
    # test_case(8, 512, 12288, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(8, 2048, 64, 'int8', bmm_s8t_s8n_s32t)
    # test_case(8, 2048, 64, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(8, 2048, 1024, 'int8', bmm_s8t_s8n_s32t)
    # test_case(8, 2048, 1024, 'int8', bmm_s8t_s8n_s32t_cublas)
    # test_case(16, 512, 12288, 'int8', bmm_s8t_s8n_s32t)
    # test_case(16, 512, 12288, 'int8', bmm_s8t_s8n_s32t_cublas)
    
# 
