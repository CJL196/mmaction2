# import tlib
import json
## 静态图片
#from d2l
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import argparse

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None, save:str = None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    if save is not None:
        plt.savefig(save)

def json_drawer(path:str, save_dir:str='.', save_prefix:str=''):
    """
    标准数据格式
    训练
    {"lr": 0.01, "data_time": 0.01911295255025228, "grad_norm": 3.746343970298767, "loss": 3.0653598149617514, "top1_acc": 0.0, "top5_acc": 0.0, "loss_cls": 3.0653598149617514, "time": 0.4302546501159668, "epoch": 1, "iter": 15, "memory": 8164, "step": 15}
    测试
    {"acc/top1": 0.0, "acc/top5": 0.14035087719298245, "acc/mean1": 0.0, "data_time": 0.047061145305633545, "time": 0.1819170117378235, "step": 1}
    """
    train_drawer = ['loss']
    test_drawer = ['ml_acc/cMAP', 'ml_acc/f1_score']
    train_arr, test_arr = [[] for i in train_drawer], [[] for i in test_drawer]
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if 'lr' in data: # is train
                for i, key in enumerate(train_drawer):
                    train_arr[i].append(data[key])
            else: # is test
                for i, key in enumerate(test_drawer):
                    test_arr[i].append(data[key])
    
    
    # plot loss
    # print(len(list(range(len(train_arr)))))
    # print(len(test_arr[2]))
    
    plot(list(range(len(train_arr[0]))), train_arr[0], save=f'{save_dir}/{save_prefix}_train_loss.png', legend=[train_drawer[0]])
    # plot train acc
    # plot(list(range(len(train_arr[0]))), train_arr[1:], save=f'{save_dir}/{save_prefix}_train_acc.png', legend=train_drawer[1:])
    # plot test acc
    plot(list(range(len(test_arr[0]))), test_arr, save=f'{save_dir}/{save_prefix}_test_acc.png', legend=test_drawer)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to json file')
    parser.add_argument('--save_dir', type=str, default='.', help='path to save dir')
    parser.add_argument('--save_prefix', type=str, default='', help='prefix of save file')
    return parser.parse_args()

def main():
    args = parser = parse_args()
    json_drawer(args.path, args.save_dir, args.save_prefix)
    

if __name__ == "__main__":
    main()