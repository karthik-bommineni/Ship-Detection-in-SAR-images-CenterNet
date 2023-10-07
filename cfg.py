# model.py
CHECKPOINT = True              # True断点保存
NET = 'DlaNet'                  # DlaNet
NUM_CLASSES = 20                 # @
LAYER = 34

# dataset.py
DATASET_NAME = 'RVOC2007'        # @
IMG_EXT = 'jpg'
# HRSC 和 UCAS-AOD的均值和方差
#MEAN = [0.5194416012442385, 0.5378052387430711,0.533462090585746]     # @
#STD = [0.3001546018824507, 0.28620901391179554, 0.3014112676161966]   # @
# RVOC2007的均值和方差
MEAN = [0.471, 0.448, 0.408]
STD = [0.234, 0.239, 0.242]

# loss.py: smooth
Loss = 'l1'           # OR 'gwd'
# train
GPU_ID = '1'
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
learning_rate = 1.25e-4
NUM_EPOCHS = 800           # RVCO需要训练时间久一点。 150

# predict
RET_IMG = 'img_ret'        # 可视化结果保存的位置
CON_SCORE = 0.3

'''
在post_process中，预测值是128*128的[cx,cy,h,w,ang]格式，若采用该格式
直接变回原图尺度，会偶尔出现w出现负值的情况。
因此，需要将[cx,cy,h,w,ang] --> [lx,ly,rx,ry,ang]映射回原图尺度，
在将其变成[cx,cy,h,w,ang]就不会出现该问题了。
'''