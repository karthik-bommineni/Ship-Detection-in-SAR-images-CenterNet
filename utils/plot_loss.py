import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mAP_dla.csv')

plt.plot(df.epoch, df.best_test_loss, label='Best Training Loss')
plt.plot(df.epoch, df.validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('../plots/dla_loss.jpg')

plt.legend()
plt.show()

