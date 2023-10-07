from mAP.map_func import eval_mAP
# 各类的ap结果存在了output.txt中。
# 将输出结果保存成
mAP = eval_mAP('./test', use_07_metric = True)
print(mAP)
