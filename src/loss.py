import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

class AGLoss(nn.Module):
	def __init__(self):
		super(AGLoss, self).__init__()

	def forward(self, gender_preds, gender_targets):
	# def forward(self, age_preds, age_targets, gender_preds, gender_targets):
		"""Compute loss between (age_preds, age_targets) and (gender_preds, gender_targets)"""
		# age_prob = F.softmax(age_preds, dim=1).int().cpu()
		# age_expect = torch.sum(Variable(torch.arange(1, 117).int())*age_prob, 1).int().cpu()
		# age_loss = F.smooth_l1_loss(age_expect, age_targets, size_average=False)
		gender_loss = F.binary_cross_entropy_with_logits(gender_preds.float().cuda(), gender_targets.float().cuda())
		# print("age_loss: %.3f | gender_loss: %.3f" & (age_loss.data[0], gender_loss.data[0]),
		# 	end='|')
		print("gender_loss: {}".format(gender_loss.data[0]))
		# return age_loss + gender_loss
		return gender_loss