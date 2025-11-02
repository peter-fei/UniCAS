import argparse
import os
import yaml

def set_type(x,target_type):
    x = list(map(target_type,x.split(',')))
    return x  

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--loss-weights',type=lambda x: set_type(x,float), default=1)
parser.add_argument('--lr_head', type=lambda x: set_type(x,float), default=0.00005)

parser.add_argument('--weights', type=str, default='',
                    help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--where', default="./ckpts", help='where to save the model')
parser.add_argument('--base_path', default='/public_bme/data/jianght/datas/Pathology/class2', help='base path for the csv files')
parser.add_argument('--train-csv', default='train.csv')
parser.add_argument('--valid-csv', '--test_neibu_csv', default='valid.csv')
parser.add_argument('--test-csv','--test_waibu_csv', default='test.csv')



parser.add_argument('--img-batch', default=800, type=int,help=' patch numbers of a sample')
parser.add_argument('--num_tasks', type=int,default=1,help='num of tasks')
parser.add_argument('--classes', type=lambda x: set_type(x,int), default=2 ,help='an integer or a list of integers')
parser.add_argument('--tasks', type= lambda x: set_type(x,str), default='label' ,help='a str or a list of tasks')

parser.add_argument('--cont', action='store_true',help='need to count high positive or not')

parser.add_argument('--encoder', default='unicas')
parser.add_argument('--embed_dim',type=int, help='embedd dim for encoder')
parser.add_argument('--aggregator', default='multi_task', help='Choice of the aggregator')
parser.add_argument('--num_experts',type=int,default=8, help='number of experts')
parser.add_argument('--depth', type=int,default=8,help='depth of the multi-task aggregator')
parser.add_argument('--capacity_factors', type=lambda x: set_type(x,int), help='number of M for task-specific attention')


parser.add_argument("--local-rank","--local_rank", help="local device id on current node",type=int,default=None)
parser.add_argument('--task_id', type=int,default=None)

parser.add_argument('--reduction', default='mean',choices=['mean','sum','none'])

parser.add_argument('--eval-only', action='store_true',help='eval mode')
parser.add_argument('--model', default='moe')
parser.add_argument('--config', default=None)
parser.add_argument('--save_pred', action='store_true', help='whether save pred result or not')


def init_args(args):
    if args.config is not None:
        assert os.path.isfile(args.config),f'Config {args.config} not exist-----------------------------------------------'
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            for k,v in config.items():
                setattr(args,k,v)

    check_attrs = ['classes','tasks']
    for attr in check_attrs:
        val = getattr(args,attr)
        assert isinstance(val, (int, float, str,list)) , f'expect type of {attr} in [int,float,str] ,but get {val} {type(val)}'

        if isinstance(val,list):
            if len(val) == 1:
                val *= args.num_tasks
                setattr(args,attr,val)
            else:
                assert len(val) == args.num_tasks, f'expect len of {attr} to be {args.num_tasks} ,but get {len(val)}'
            continue
        else:
            val = [val] * args.num_tasks
            setattr(args,attr,val)
            
    args.train_csv = os.path.join(args.base_path,args.train_csv)
    args.valid_csv = os.path.join(args.base_path,args.valid_csv)
    args.test_csv = os.path.join(args.base_path,args.test_csv)
    
    if args.task_id is None:
        args.task_id = [i for i in range(args.num_tasks) if args.lr_head[i]>1e-8]  # if lr < 1e-8, do not train this task
    elif args.num_tasks == 1:
        args.task_id = [args.task_id]

    return args

    
def get_args():
    args = parser.parse_args()
    init_args(args)

    return args

if __name__ == '__main__':
    args = get_args()
    print(args)

