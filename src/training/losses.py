import torch
import torch.nn as nn

from . import base
from . import functional as F
from . import _modules as modules


class JaccardLoss(base.Loss):

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = modules.Activation(activation, dim=1)
        self.per_image = per_image
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(base.Loss):

    def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, class_weights=None, logits=False, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None]
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        if self.logits:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pr, gt, reduction='none')
        else:
            bce_loss = F.binary_crossentropy(pr, gt, label_smoothing=self.label_smoothing)

        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()

        return focal_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.focal(y_pred, y_true) + self.dice(y_pred, y_true)


class BCEDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.bce(y_pred, y_true) + self.dice(y_pred, y_true)


import re
import torch.nn as nn

class BaseObject(nn.Module):
  def __init__(self, name=None):
    super().__init__()
    self._name = name

  @property
  def __name__(self):
    if self._name:
      return self._name

    name = self.__class__.__name__
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class Metric(BaseObject):
  pass

class Loss(BaseObject):
  def __add__(self, other):
    if isinstance(other, Loss):
      return SumOfLosses(self, other)
    else:
      raise ValueError('Loss should be inherited from `Loss` class')

  def __radd__(self, other):
    return self.__add__(other)

  def __mul__(self, other):
    if isinstance(other, Loss):
      return MultipliedLoss(self, other)
    else:
      raise ValueError('Loss should be inherited from `Loss` class')

  def __rmul__(self, other):
    return self.__mul__(other)

class SumOfLoss(Loss):
  def __init__(self, l1, l2):
    name = f'{l1.__name__} + {l2.__name__}'
    super().__init__(name=name)
    self.l1 = l1
    self.l2 = l2

  def forward(self, *inputs):
    return self.l1(*inputs) + self.l2(*inputs)

class MultipliedLoss(Loss):
  def __init__(self, loss, multiplier):
    if len(loss.__name__.split('+')) > 1:
      name = f'{multiplier} * ({loss.__name__})'
    else:
      name = f'{multiplier} + {loss.__name__}'
    super().__init__(name=name)
    self.loss = loss
    self.multiplier = multiplier

  def forward(self, *inputs):
    return self.multiplier * self.loss(*inputs)

class QuadraticLoss(Loss):
  def __init__(self, activation=None, ignore_channels=None, per_image=False, **kwargs):
    super().__init__(**kwargs)
    self.activation = modules.Activation(activation, dim=1)
    self.loss = nn.MSELoss(reduction='mean')

  def forward(self, y_pr, y_gt):
    ypr = self.activation(y_pr)
    return self.loss(y_pr, y_gt)