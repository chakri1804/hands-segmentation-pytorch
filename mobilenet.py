from torch import nn
from torchvision.models import mobilenet_v2

class CustomMobileNetv2(nn.Module):
	def __init__(self, output_size):
		super().__init__()
		self.mnet = mobilenet_v2(pretrained=True)
		# self.freeze()
		self.classifier = nn.Linear(1000, output_size)
		self.softmax = nn.Softmax()
		# self.mnet.classifier = nn.Sequential(
		# 	nn.Linear(1280, output_size),
		# 	nn.LogSoftmax(1)
		# )

	def forward(self, x):
		x = self.mnet(x)
		x = self.softmax(self.classifier(x))
		return x
  
	def freeze(self):
		for param in self.mnet.parameters():
			param.requires_grad = False

	def unfreeze(self):
		for param in self.mnet.parameters():
			param.requires_grad = True
