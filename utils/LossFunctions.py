from keras import backend as K

def jaccard_loss(y_true, y_pred):
    smooth = 0.001
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = 1 - (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def iou_metric(y_true, y_pred):
    smooth = 0.001
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(iou)

def jaccard_loss_weighted(y_true, y_pred, weights = 1):
    smooth = 0.001
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = 1 - (intersection + smooth) / (sum_ - intersection + smooth)
    jac = jac*weights
    return K.mean(jac)