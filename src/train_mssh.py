import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from model import getModel, saveModel
from datasets.mpii import MPII
from utils.logger import Logger
from train import train, val
import scipy.io as sio
from models.mssh import MSSH
from utils.utils import AverageMeter, Flip, ShuffleLR
from utils.eval import Accuracy, getPreds, finalPreds
import cv2
import ref
from progress.bar import Bar
from utils.debugger import Debugger
def main():
  opt = opts().parse()
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir, now.isoformat())
  model = MSSH().cuda()
  optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                    alpha = ref.alpha, 
                                    eps = ref.epsilon, 
                                    weight_decay = ref.weightDecay, 
                                    momentum = ref.momentum)
  criterion = torch.nn.MSELoss().cuda()

  # if opt.GPU > -1:
  #   print('Using GPU', opt.GPU)
  #   model = model.cuda(opt.GPU)
  #   criterion = criterion.cuda(opt.GPU)



  val_loader = torch.utils.data.DataLoader(
      MPII(opt, 'val'), 
      batch_size = 1, 
      shuffle = False,
      num_workers = int(ref.nThreads)
  )

  if opt.test:
    log_dict_train, preds = val(0, opt, val_loader, model, criterion)
    sio.savemat(os.path.join(opt.saveDir, 'preds.mat'), mdict = {'preds': preds})
    return

  train_loader = torch.utils.data.DataLoader(
      MPII(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True if opt.DEBUG == 0 else False,
      num_workers = int(ref.nThreads)
  )

  for epoch in range(1):
    model.train()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []
  
    nIters = len(train_loader)
    bar = Bar('{}'.format(opt.expID), max=nIters)
    for i, (input, target, meta) in enumerate(train_loader):
      input_var = torch.autograd.Variable(input).float().cuda()
      target_var = torch.autograd.Variable(target).float().cuda()
      #print( input_var)
      output = model(input_var)
    
      loss = criterion(output, target_var)
      Loss.update(loss.data[0], input.size(0))
      Acc.update(Accuracy((output.data).cpu().numpy(), (target_var.data).cpu().numpy()))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = "train")
      bar.next()

    bar.finish()


if __name__ == '__main__':
  main()
