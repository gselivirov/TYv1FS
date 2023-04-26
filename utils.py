import json
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from collections import Counter

import torch

#Data
#-------
def xml_to_yolo_bbox(bbox: List, w: int, h: int) -> List:
    """
    Convert bouniding boxes in PASCAL VOC format ([xmin, ymin, xmax, ymax]) to YOLO format ([x_center, y_center, width, heigth])

    Args:
        bbox (List): bounding box [xmin, ymin, xmax, ymax]
        w: width of the image
        h: height of the image

    Returns:
        [x_center, y_center, width, heigth]
    """

    x_center = ((bbox[0] + bbox[2]) / 2) / w
    y_center = ((bbox[1] + bbox[3]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox: List, w: int, h: int) -> List:
    """
    Convert bounding boxes in YOLO format ([x_center, y_center, width, heigth]) to PASCAL VOC format ([[xmin, ymin, xmax, ymax]])

    Args: 
        bbox (List): bounding box [x_center, y_center, width, heigth]
        w: width of the image
        h: height of the image
    """

    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def convert_annotations_to_yolo(
        classes: List, 
        input_dir: Path, 
        output_dir: Path, 
        image_dir: Path):
    """
    Convert xml files with annotations to txt files in YOLO (midpoint) format

    Args:
        classes (List[str]): list of classes
        input_dir: path to the directory with xml labels
        output_dir: desired path to the output
        image_dir: path to the directory with images
    """
    
    # create the labels folder (output directory)
    output_dir.mkdir(exist_ok=True)

    # identify all the xml files in the annotations folder (input directory)
    annotations = input_dir.glob('*.xml')
    # loop through each 
    for annotation in annotations:
        filename = annotation.stem
        # check if the label contains the corresponding image file
        if not (image_dir / f'{filename}.png').exists(): 
            print(f'{filename} image does not exist!')
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(annotation)
        root = tree.getroot()
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        for obj in root.findall('object'):
            label = obj.find('name').text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find('bndbox')]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = ' '.join([str(x) for x in yolo_bbox])
            result.append(f'{index} {bbox_string}')

        if result:
            with open((output_dir / f'{filename}.txt'), 'w', encoding='utf-8') as f:
            # generate a YOLO format text file for each xml file
                f.write('\n'.join(result))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))

    print(f'Created {len(list(output_dir.iterdir()))} txt annotations. \nClasses: {classes}')



#Loss and stuff
#--------------

def intersection_over_union(
        pred_boxes: torch.Tensor, 
        label_boxes: torch.Tensor, 
        box_format: str = 'midpoint') -> torch.Tensor:
	'''
	Calculates intersection over union for a batch of bounding boxes in corner or midpoint format.
	
	Args: 
		pred_boxes (torch.Tensor): Predicted bounding boxes with shape (N, 4) where N is BATCH_SIZE. 
		label_boxes (torch.Tensor): True bounding boxes with shape (N, 4) where N is BATCH_SIZE.
		box_format (str): coreners (x1, y1, x2, y2) or midpoint (x, y, w, h)
	
	Returns: 
		torch.Tensor: Tensor of shape (N, 1) with intersection over union for all elements of a batch. 
	'''

	# get the coordinates of both bounding boxes
	if box_format == "midpoint":
		x1_pred = pred_boxes[..., 0:1] - pred_boxes[..., 2:3] / 2
		y1_pred = pred_boxes[..., 1:2] - pred_boxes[..., 3:4] / 2
		x2_pred = pred_boxes[..., 0:1] + pred_boxes[..., 2:3] / 2
		y2_pred = pred_boxes[..., 1:2] + pred_boxes[..., 3:4] / 2

		x1_label = label_boxes[..., 0:1] - label_boxes[..., 2:3] / 2
		y1_label = label_boxes[..., 1:2] - label_boxes[..., 3:4] / 2
		x2_label = label_boxes[..., 0:1] + label_boxes[..., 2:3] / 2
		y2_label = label_boxes[..., 1:2] + label_boxes[..., 3:4] / 2

	if box_format == 'corners':
		x1_pred = pred_boxes[..., 0:1]
		y1_pred = pred_boxes[..., 1:2]
		x2_pred = pred_boxes[..., 2:3]
		y2_pred = pred_boxes[..., 3:4] 

		x1_label = label_boxes[..., 0:1]
		y1_label = label_boxes[..., 1:2]
		x2_label = label_boxes[..., 2:3]
		y2_label = label_boxes[..., 3:4]
	
	#calculate coordinates of intersection
	x1 = torch.max(x1_pred, x1_label)
	y1 = torch.max(y1_pred, y1_label)
	x2 = torch.max(x2_pred, x2_label)
	y2 = torch.max(y2_pred, y2_label)
	
	#calculate the area of intercection (h*w)
	intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
	
	#calculate the area of bounding boxes
	pred_area = abs((x2_pred - x1_pred) * (y2_pred - y1_pred))
	label_area = abs((x2_label - x1_label) * (y2_label - y1_label))
	
	#divide intersection by union
	return intersection / (pred_area + label_area - intersection + 1e-6)


def non_max_suppression(
        bboxes: List[List[float]],
        iou_threshold: float,
        prob_threshold: float, 
        box_format:str = 'corners') -> List[List[float]]:
    """
    Filters predicted bounding boxes within class based on probability and IoU.
    Args:
        bboxes (List[List[float]]): List of bounding boxes [[class, prob, x1, y1, x2, y2]]
        iou_threshold (float): Cutoff IoU to discard boxes
        prob_threshold (float): Cutoff probability to discard boxes regardless of IoU
        box_format (str): 'corners' or 'midpoint' (x1, y1, x2, y2) / (x, y, w, h) for intersection_over_union()
    Returns:
        List[List[float]]: List of filtered bounding boxes after non-maximum suppression.
    """
    # Discard bboxes with low probability and sort in descending order
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    boxes_after_nms = []
    # While there are boxes left,
    # take the one with the highest probability and discard all the boxes of the same class with IoU > iou_threshold
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                pred_boxes = torch.tensor(chosen_box[2:]),
                label_boxes = torch.tensor(box[2:]),
                box_format = box_format) < iou_threshold
        ]

        boxes_after_nms.append(chosen_box)

    return boxes_after_nms



def mean_average_precision(
        pred_boxes: List,
        true_boxes: List, 
        iou_threshold: float, 
        n_classes: int, 
        box_format: str = 'midpoint'):
  """
  Calculate mean of average precisions for each class at a given iou threshold
  Args:
      pred_boxes (list): [[img_idx, class, prob, x1, y1, x2, y2], ...]
      true_boxes (list): [[img_idx, class, prob, x1, y1, x2, y2], ...]
      iou_threshold (float): value of intersection over union to regard the bbox as good
      n_classes (int): number of classes
      box_format (str): format of bboxes
  """
  average_precisions = []

  #for each class
  for c in range(n_classes):
    detections = []
    ground_truths = []
    #for each predicted bboxes check if it is the class we are looking at
    for detection in pred_boxes:
      if detection[1] == c: #[1] = class
        detections.append(detection)

    #for each true bbox check if it is the class we are looking at
    for true_box in true_boxes:
      if true_box[1] == c:
        ground_truths.append(true_box)

    #create a dict {img_index: number_of_true_boxes} for the class we are currentlly looking at
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    #rewrite the dict to be {image_index: torch.Tensor(0, 0, ...n..., 0)} where n is the number of true bboxes
    for key, val in amount_bboxes.items():
      amount_bboxes[key] = torch.zeros(val)

    #sort predicted bboxes by probability score for the current class in descending order
    detections.sort(key = lambda x: x[2], reverse=True)

    #create tensors with number of elements = number of predicted bboxes for a given class
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    amount_true_boxes = len(ground_truths) # total number of true bboxes per class

    #for each predicted bounding box in current class
    for detection_idx, detection in enumerate(detections):
      #get all the true bboxes for current image
      ground_truth_img = [
          bbox for bbox in ground_truths 
          if bbox[0] == detection[0]
          ]
      #number of bboxes for the current image
      num_gts = len(ground_truth_img)
      best_iou = 0

      #calculate iou for each ground true bbox in the image (we are inside the loop for each predicted bbox in the class)
      for idx, gt in enumerate(ground_truth_img):
        iou = intersection_over_union(
            torch.tensor(detection[3:]),
            torch.tensor(gt[3:]),
            box_format
            )
        #find the best corresponding gorund truth bbox in the image
        if iou > best_iou:
          best_iou = iou
          best_gt_idx = idx
      
      #if the current prediction is good (iou > threshold)
      if best_iou > iou_threshold:
        #if we didn't cover this bbox before
        #amount_bboxes[detection[0]] dict image_idx: torch.zeros(number of true bboxes in the image) detection[0]==img_idx
        #amount_bboxes[detection[0]][best_gt_idx] best_gt_idx == target bbox index
        if amount_bboxes[detection[0]][best_gt_idx] == 0:
          #mark it as a true positive
          TP[detection_idx] = 1
          #mark it as covered
          amount_bboxes[detection[0]][best_gt_idx] = 1
        #else mark it as false positive
        else:
          FP[detection_idx] = 1
      else:
        FP[detection_idx] = 1
    
    #callculate cummulative sum of true positives for class
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)

    recalls = TP_cumsum / (amount_true_boxes + 1e-6)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    average_precisions.append(torch.trapezoid(precisions, recalls))
  
  return sum(average_precisions) / len(average_precisions)



#Work in progress
#----------------
def convert_cellboxes(predictions, S=7):
  """
  Converts bounding boxes output from Yolo with
  an image split size of S into entire image ratios
  rather than relative to cell ratios. Tried to do this
  vectorized, but this resulted in quite difficult to read
  code... Use as a black box? Or implement a more intuitive,
  using 2 for loops iterating range(S) and convert them one
  by one, resulting in a slower but more readable implementation.
  """

  predictions = predictions.to("cpu")
  batch_size = predictions.shape[0]
  predictions = predictions.reshape(batch_size, 7, 7, 13)
  bboxes1 = predictions[..., 4:8]
  bboxes2 = predictions[..., 9:13]
  scores = torch.cat(
      (predictions[..., 3].unsqueeze(0), predictions[..., 8].unsqueeze(0)), dim=0
  )
  best_box = scores.argmax(0).unsqueeze(-1)
  best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
  cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
  x = 1 / S * (best_boxes[..., :1] + cell_indices)
  y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
  w_y = 1 / S * best_boxes[..., 2:4]
  converted_bboxes = torch.cat((x, y, w_y), dim=-1)
  predicted_class = predictions[..., :3].argmax(-1).unsqueeze(-1)
  best_confidence = torch.max(predictions[..., 3], predictions[..., 9]).unsqueeze(
      -1
  )
  converted_preds = torch.cat(
      (predicted_class, best_confidence, converted_bboxes), dim=-1
  )

  return converted_preds

def cellboxes_to_boxes(out, S=7):
  converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
  converted_pred[..., 0] = converted_pred[..., 0].long()
  all_bboxes = []

  for ex_idx in range(out.shape[0]):
      bboxes = []

      for bbox_idx in range(S * S):
          bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
      all_bboxes.append(bboxes)

  return all_bboxes

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="mps",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes
        


