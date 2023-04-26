import torch
from torch import nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.labmbda_noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        iou_b1 = intersection_over_union(predictions[..., (self.C + 1):(self.C + 5)], target[..., (self.C + 1):(self.C + 5)]) #format of prediction [class_probs(0 - n_classes-1), class_score(n_classes), 4_coords+prob_obj for each bbox]
        iou_b2 = intersection_over_union(predictions[..., (self.C + 6):(self.C + 10)], target[..., (self.C + 1):(self.C + 5)])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)],  dim=0)
        best_box = torch.argmax(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)


        #Box coordinates
        box_predictions = exists_box * (
            (
            best_box * predictions[..., (self.C + 6):(self.C + 10)] +
            (1 - best_box) * predictions[..., (self.C + 1):(self.C + 5)]
            )
        )
        box_targets = exists_box * target[..., (self.C + 1):(self.C + 5)]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6) 
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )


        #Object loss
        pred_box = (
            best_box * predictions[..., (self.C + 5):(self.C + 6)] +
            (best_box - 1) * predictions[..., (self.C):(self.C + 1)]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:(self.C + 1)])
        )


        #No object loss
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:(self.C + 1)], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:(self.C + 1)], start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., (self.C + 5):(self.C + 6)], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:(self.C + 1)], start_dim=1)
        )


        #Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss 
            + object_loss 
            + self.labmbda_noobj * no_obj_loss 
            + class_loss
        )

        return loss