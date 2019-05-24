import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.autograd import Function
import io


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

revgrad = RevGrad.apply

class RevGrad(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return revgrad(input_)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = models.resnet18(pretrained=True)
        self.convnet_layer = self.convnet._modules.get('avgpool')
        self.convnet.eval()

        self.fc = nn.Sequential(nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 30)
                                )

    def forward(self, x):
        my_embedding = torch.cuda.FloatTensor(x.shape[0],512,1,1).fill_(0)

        def copy_data(m,i,o):
            my_embedding.copy_(o.data)

        h = self.convnet_layer.register_forward_hook(copy_data)

        self.convnet(x)
        h.remove()

        my_embedding = my_embedding.view(my_embedding.size()[0], -1)
        output = self.fc(my_embedding)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class Resnet152EmbeddingNet(nn.Module):
    def __init__(self, dim=64):
        super(Resnet152EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 1024),
                                nn.PReLU(),
                                nn.Linear(1024, 512),
                                nn.PReLU(),
                                nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, dim))


    def forward(self, x):
        return self.fc(x)

    def get_embedding(self, x):
        return self.forward(x)

class Resnet18EmbeddingNet(nn.Module):
    def __init__(self, dim=64):
        super(Resnet18EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, dim))

    def forward(self, x):
        return self.fc(x)

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TextEmbeddingNet(nn.Module):
    def __init__(self, dim=64):
        super(TextEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(300, 256),
                nn.PReLU(),
                nn.Linear(256, 256),
                nn.PReLU(),
                nn.Linear(256, 128),
                nn.PReLU(),
                nn.Linear(128, dim))


    def forward(self, x):
        return self.fc(x)

class TwoStreamVideoEmbeddingNet(nn.Module):
    def __init__(self):
        super(TwoStreamVideoEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2560, 1024),
                                 nn.PReLU(),
                                 nn.Linear(1024, 512),
                                 nn.PreLU(),
                                 nn.Linear(512, 256),
                                 nn.PreLU(),
                                 nn.Linear(256, 64))

    def forward(self, spatial_feat, temporal_feat):
        video_embedding = torch.cat((spatial_feat, temporal_feat), dim=1)
        return self.fc(video_embedding)

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class IntermodalTripletNet(nn.Module):

    def __init__(self, modalityOne_net, modalityTwo_net):
        super(IntermodalTripletNet, self).__init__()
        self.modalityOneNet = modalityOne_net
        self.modalityTwoNet = modalityTwo_net

    def forward(self, a_v, p_t, n_t, a_t, p_v, n_v):
        output_anch1 = self.modalityOneNet(a_v)
        output_pos2 = self.modalityTwoNet(p_t)
        output_neg2 = self.modalityTwoNet(n_t)

        output_anch2 = self.modalityTwoNet(a_t)
        output_pos1 = self.modalityOneNet(p_v)
        output_neg1 = self.modalityOneNet(n_v)

        return output_anch1, output_pos2, output_neg2, output_anch2, output_pos1, output_neg1

    def get_modOne_embedding(self, x):
        return self.modalityOneNet(x)

    def get_modTwo_embedding(self, x):
        return self.modalityTwoNet(x)

class ModalityDiscriminator(nn.Module):

    def __init__(self, dim=64):
        super(ModalityDiscriminator, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(dim, 64),
                nn.PReLU(),
                nn.Linear(64, 1))

    def forward(self, embedding):
        return F.softmax(self.fc(embedding), dim=0)

class FeatureExtractor(nn.Module):
    def __init__(self, net):
        super(FeatureExtractor, self).__init__()
        if net == "resnet152":
            net = models.resnet152(pretrained=True)
            dim = 2048
        elif net == "resnet18":
            net = models.resnet18(pretrained=True)
            dim = 2048
        else:
            raise RuntimeException("'{}' not supported".format(net))
        self.net = net.eval()
        self.dim = dim
        self.penult_layer = self.net._modules.get('avgpool')
    
    def forward(self, x):
        output = self.get_embedding(self, x)
        return output
    
    def get_embedding(self, x):
        embedding = torch.cuda.FloatTensor(x.shape[0], self.dim, 1, 1).fill_(0)
        def copy(m, i ,o):
            embedding.copy_(o.data)
        hook = self.penult_layer.register_forward_hook(copy)
        self.net(x)
        hook.remove()
        return embedding.view(embedding.size()[0], -1)
