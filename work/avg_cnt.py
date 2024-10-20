import json
## 静态图片
#from d2l

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to json file')
    parser.add_argument('--type', type=str, required=True, choices=['train', 'test'], help='type of json file')
    parser.add_argument('--legend', type=str, required=True)
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, help='end index')
    
    return parser.parse_args()

def json_reader(path:str, type:str, start:int, end:int, legend:str):
    if type not in ['train', 'test']:
        raise ValueError("type must be 'train' or 'test'")
    train_drawer = ['loss', 'top1_acc', 'top5_acc']
    test_drawer = ['acc/top1', 'acc/top5', 'acc/mean1']
    train_arr, test_arr = [[] for i in train_drawer], [[] for i in test_drawer]
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if type == 'train' and 'lr' in data:
                for i, key in enumerate(train_drawer):
                    train_arr[i].append(data[key])
            if type == 'test' and 'lr' not in data:
                for i, key in enumerate(test_drawer):
                    test_arr[i].append(data[key])
    if type == 'train':
        idx = train_drawer.index(legend)
        ans = sum(train_arr[idx][start:end]) / (end - start)
        print(ans)
    else:
        idx = test_drawer.index(legend)
        ans = sum(test_arr[idx][start:end]) / (end - start)
        print(ans)

def main():
    args = parser = parse_args()
    json_reader(args.path, args.type, args.start, args.end, args.legend)
    

if __name__ == "__main__":
    main()