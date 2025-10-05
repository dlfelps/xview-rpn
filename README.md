The goal of this project is to train a region proposal network (RPN) on the xView dataset. It will be trained in the context of a pretrained faster rcnn network from the torchvision python package. 

The default configuration will be altered in the following ways:
1. The backbone will be frozen
2. the detection heads will be frozen (class attributes from xView will only be used to differentiate objects from background)
3. the minimum size input image will be set to 224 
4. the anchor sizes will be [16, 32, 64]
5. The aspect ratios will be [0.5, 1.0, 2.0]

To successfully complete this task we will also need a data wrapper for xView that clips large images into 224 x 224 (along with the relative object locations determined by the ground truth GeoJSON file). Modify the loss function of the RPN to ignore the ROI head loss and focus on the RPN loss.

rpn_loss = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']

RPN-only evaluation: Measure recall of proposals vs. ground truth



