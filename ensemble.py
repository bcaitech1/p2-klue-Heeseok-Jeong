import numpy as np
import pandas as pd

logit_path = '/opt/ml/code/prediction/logits/'
model_names = []
model_names.append('xlmroberta7_checkpoint-3000.npy')
# model_names.append('xlmroberta8_checkpoint-3000.npy')
model_names.append('xlmroberta10_checkpoint-2000.npy')
model_names.append('xlmroberta11_checkpoint-2500.npy')
model_names.append('koelectra2_checkpoint-2500.npy')
model_names.append('bert1_checkpoint-3500.npy')
logits_sum = np.array([[0 for _ in range(42)] for _ in range(1000)], np.float64)
for model_name in model_names:
    model_logits = np.load(logit_path + model_name)
    logits_sum += model_logits
logits_mean = logits_sum / len(model_names)
additional = np.array([0.1] + [0 for _ in range(41)], np.float64)
logits_mean += additional
logits = logits_mean.argmax(-1)
# print(logits[:10])

logits_pd = pd.DataFrame(logits)
# logits_pd.rename(columns={'0':'pred'}, inplace=True)
# for col in logits_pd.columns:
#     print(col)

logits_pd.rename(columns={0:'pred'}, inplace=True)
print(logits_pd)
output_dir = '/opt/ml/code/prediction/'
output_dir += 'ensemble3'
output_dir += '.csv'
logits_pd.to_csv(output_dir, index=False)
print(f"{output_dir} file saved!")
