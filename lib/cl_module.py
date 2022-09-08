import torch
import torch.nn.functional as F
import torch.nn as nn

def simclr_loss(embedding1, embedding2, cls_target, logit_scale):
    positive_idxs = torch.nonzero(cls_target).squeeze(1)

    embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)

    #mask (simCLR)
    mask = torch.eye(embedding1.size(0), device = embedding1.device) * 1e9
    q_a = embedding1
    q_b = embedding2

    logits_aa = torch.matmul(q_a, q_a.transpose(0, 1))
    logits_aa = logits_aa - mask
    logits_bb = torch.matmul(q_b, q_b.transpose(0, 1)) 
    logits_bb = logits_bb - mask
    logits_ab = torch.matmul(q_a, q_b.transpose(0, 1)) 
    logits_ba = torch.matmul(q_b, q_a.transpose(0, 1))

    targets = torch.arange(logits_aa.size()[0], device = embedding1.device)

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), targets)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), targets)
    loss = (loss_a + loss_b) / 2 
    return loss

def NCE_loss(text_embedding, video_embedding, cls_target, logit_scale, only_positive=True):
    if only_positive:
        # only select idxs with positive labels
        cls_target = cls_target.bool()
        text_embedding = text_embedding[cls_target]
        video_embedding = video_embedding[cls_target]

    # normalize embeddings
    text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)
    video_embedding = video_embedding / video_embedding.norm(dim=1, keepdim=True)

    logit_scale = logit_scale.exp()
    logits = logit_scale * text_embedding @ video_embedding.t()
    targets = torch.arange(logits.size()[0], device = logits.device)
    loss = F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    return loss / 2.0


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class cl_MLP(torch.nn.Module):
    def __init__(self,
                 in_sizes,
                 out_sizes,
                 activation,
                 output_activation='identity',
                 **kwargs
                 ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]
        layers = []
        for in_size, out_size in zip(in_sizes, out_sizes):
            layers.append(nn.Linear(in_size, out_size))
            # layers.append(nn.LayerNorm(out_size)) NOTE: empirically this does not work well, on small training set
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(activation)

        if activation != output_activation:
            layers[-1] = output_activation  # replace last activation with output activation

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)