'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.externals import joblib

def process_stdin(stdin):
    return stdin.decode('utf-8').replace('\n', '')

def save_checkpoint(model, acc, epoch, name):
    state = {
        'model': model.module if args.cuda else model,
        'acc': acc,
        'epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f"./checkpoint/{name}")

def get_model_name(model):
    return str(model).replace('\n', '').replace(' ', ''),

def save_model(model, accuracy, loss, epoch, args):
    base_name = get_model_name(model)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    if args.git_hash:
        git_hash = process_stdin(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]))
    else:
        git_hash = 'test'
    user = process_stdin(subprocess.check_output(["whoami"]))
    model_identifier = f"{base_name}-{timestamp}-{git_hash}"
    model_name = f'{model_identifier}.torch'
    with open('stats/model-stats.csv', 'a') as model_log_file:
        hyper_params = f"{args.batch_size},{args.epochs},{args.lr},"\
                       f"{args.momentum},{args.seed}"
        result_line = f'{model_name},{user},{hyper_params},{accuracy:.2f},{loss:.2f}\n'
        model_log_file.write(result_line)
    torch.save(model, model_name)
    torch.save(model, "model.torch")
    #plot_roc_curve(model, X_test, y_test, model_identifier)
    #plot_confusion_matrix(confusion_matrix, ['Annual Giver', 'Major Donor'],
    #                      model_identifier,
    #                      title='Confusion matrix')

def plot_roc_curve(model, X_test, y_test, identifier):
    y_test_probs = [p[1] for p in model.predict_proba(X_test)]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_probs)
    plt.clf()
    plt.plot(fpr, tpr, color='b')
    plt.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11), color='r')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.savefig(f'model/plots/roc-curve-{identifier}.png', dpi=100)

def plot_confusion_matrix(cm, classes, identifier,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'model/plots/confusion-matrix-{identifier}.png', dpi=100)



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
