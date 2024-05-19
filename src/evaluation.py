import torch
import torch.nn.functional as F



def precision_score_(groundtruth_mask, pred_mask):
    """
    Compute the precision score.
    
    Args:
    groundtruth_mask (torch.Tensor): Ground truth mask.
    pred_mask (torch.Tensor): Predicted mask.
    
    Returns:
    precision (float): Precision score.
    """
    intersect = torch.sum(pred_mask * groundtruth_mask).float()
    total_pixel_pred = torch.sum(pred_mask).float()
    precision = intersect / total_pixel_pred
    return round(precision.item(), 3)


def recall_score_(groundtruth_mask, pred_mask):
    """
    Compute the recall score.
    
    Args:
    groundtruth_mask (torch.Tensor): Ground truth mask.
    pred_mask (torch.Tensor): Predicted mask.
    
    Returns:
    recall (float): Recall score.
    """
    intersect = torch.sum(pred_mask * groundtruth_mask).float()
    total_pixel_truth = torch.sum(groundtruth_mask).float()
    recall = intersect / total_pixel_truth
    return round(recall.item(), 3)


def accuracy(groundtruth_mask, pred_mask):
    """
    Compute the accuracy.
    
    Args:
    groundtruth_mask (torch.Tensor): Ground truth mask.
    pred_mask (torch.Tensor): Predicted mask.
    
    Returns:
    acc (float): Accuracy.
    """
    intersect = torch.sum(pred_mask * groundtruth_mask).float()
    union = torch.sum(pred_mask).float() + torch.sum(groundtruth_mask).float() - intersect
    xor = torch.sum(groundtruth_mask == pred_mask).float()
    acc = xor / (union + xor - intersect)
    return round(acc.item(), 3)


# Example ground truth and predicted masks (for a single example)
# groundtruth_mask = torch.tensor([[0, 1, 1], [0, 1, 0], [1, 1, 1]])
# pred_mask = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 1, 0]])

# # Calculate metrics
# precision = precision_score_(groundtruth_mask, pred_mask)
# recall = recall_score_(groundtruth_mask, pred_mask)
# acc = accuracy(groundtruth_mask, pred_mask)

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"Accuracy: {acc}")


def evaluate_metrics(test_images, test_masks, model, device):
    """
    Evaluate precision, recall, and accuracy for each image and average them.
    
    Args:
    test_images (List[torch.Tensor]): List of test images, each of shape (channels, height, width).
    test_masks (List[torch.Tensor]): List of ground truth masks, each of shape (height, width).
    model (torch.nn.Module): Trained segmentation model.
    device (torch.device): Device to perform computation on ('cpu' or 'cuda').
    
    Returns:
    avg_precision (float): Average precision score.
    avg_recall (float): Average recall score.
    avg_accuracy (float): Average accuracy.
    """
    precision_list = []
    recall_list = []
    accuracy_list = []

    model.eval()
    
    with torch.no_grad():
        for test_img, img_mask in zip(test_images, test_masks):
            test_img = test_img.to(device)
            img_mask = img_mask.to(device)
            predicted_img = model(test_img.unsqueeze(0)).squeeze(0)
            predicted_img = torch.argmax(predicted_img, dim=0)

            precision = precision_score_(img_mask, predicted_img)
            recall = recall_score_(img_mask, predicted_img)
            acc = accuracy(img_mask, predicted_img)

            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(acc)
    
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average Accuracy: {avg_accuracy}")

    return avg_precision, avg_recall, avg_accuracy

# Evaluate on test set
avg_precision, avg_recall, avg_accuracy = evaluate_metrics(test_images, test_masks, model, device)


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Coefficient.
    
    Args:
    y_true (torch.Tensor): Ground truth mask.
    y_pred (torch.Tensor): Predicted mask.
    smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
    dice (float): Dice Coefficient.
    """
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def evaluate_dice_score(test_images, test_masks, model, n_classes, device):
    """
    Evaluate Dice Score for each class and average them.
    
    Args:
    test_images (List[torch.Tensor]): List of test images, each of shape (channels, height, width).
    test_masks (List[torch.Tensor]): List of ground truth masks, each of shape (height, width).
    model (torch.nn.Module): Trained segmentation model.
    n_classes (int): Number of classes.
    device (torch.device): Device to perform computation on ('cpu' or 'cuda').

    Returns:
    avg_dice_scores (List[float]): List of average Dice scores for each class.
    """
    dfs = {i: [] for i in range(n_classes)}
    model.eval()
    
    with torch.no_grad():
        for test_img, img_mask in zip(test_images, test_masks):
            test_img = test_img.to(device)
            img_mask = img_mask.to(device)
            predicted_img = model(test_img.unsqueeze(0)).squeeze(0)
            predicted_img = torch.argmax(predicted_img, dim=0)

            img_mask_exp = F.one_hot(img_mask, num_classes=n_classes).permute(2, 0, 1)
            img_pred_exp = F.one_hot(predicted_img, num_classes=n_classes).permute(2, 0, 1)
            
            for i in range(n_classes):
                df = dice_coef(img_mask_exp[i], img_pred_exp[i]).item()
                dfs[i].append(df)
    
    avg_dice_scores = [sum(dfs[i]) / len(dfs[i]) for i in range(n_classes)]
    
    for i, avg in enumerate(avg_dice_scores):
        print(f"Dice score of class {i}: {avg}")

    return avg_dice_scores


def mean_iou(y_true, y_pred, num_classes):
    """
    Compute the Mean Intersection over Union (IoU).
    
    Args:
    y_true (torch.Tensor): Ground truth mask.
    y_pred (torch.Tensor): Predicted mask.
    num_classes (int): Number of classes.
    
    Returns:
    iou (float): Mean IoU.
    """
    y_true = y_true.contiguous().view(-1)
    y_pred = y_pred.contiguous().view(-1)

    ious = []
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        intersection = (true_cls & pred_cls).sum().float().item()
        union = (true_cls | pred_cls).sum().float().item()
        if union == 0:
            ious.append(float('nan'))  # To avoid division by zero
        else:
            ious.append(intersection / union)
    
    return torch.tensor(ious).nanmean().item()

def evaluate_iou(test_images, test_masks, model, n_classes, device):
    """
    Evaluate IoU for each class and average them.
    
    Args:
    test_images (List[torch.Tensor]): List of test images, each of shape (channels, height, width).
    test_masks (List[torch.Tensor]): List of ground truth masks, each of shape (height, width).
    model (torch.nn.Module): Trained segmentation model.
    n_classes (int): Number of classes.
    device (torch.device): Device to perform computation on ('cpu' or 'cuda').

    Returns:
    avg_iou (float): Average IoU.
    """
    ious = []
    model.eval()
    
    with torch.no_grad():
        for test_img, img_mask in zip(test_images, test_masks):
            test_img = test_img.to(device)
            img_mask = img_mask.to(device)
            predicted_img = model(test_img.unsqueeze(0)).squeeze(0)
            predicted_img = torch.argmax(predicted_img, dim=0)

            img_mask_exp = F.one_hot(img_mask, num_classes=n_classes).permute(2, 0, 1)
            img_pred_exp = F.one_hot(predicted_img, num_classes=n_classes).permute(2, 0, 1)
        
            iou = mean_iou(img_mask, predicted_img, n_classes)
            ious.append(iou)
    
    avg_iou = sum(ious) / len(ious)
    print(f"Average IoU: {avg_iou}")

    return avg_iou

#-----------------------------------------------#

# Usage Example 

# import torch

# # Define device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Assume test_images and test_masks are lists of torch.Tensor, and model is a trained torch.nn.Module
# n_classes = 9

# # Move model to device
# model.to(device)

# # Evaluate Dice Score
# dice_scores = evaluate_dice_score(test_images, test_masks, model, n_classes, device)

# # Evaluate IoU
# iou_score = evaluate_iou(test_images, test_masks, model, n_classes, device)
