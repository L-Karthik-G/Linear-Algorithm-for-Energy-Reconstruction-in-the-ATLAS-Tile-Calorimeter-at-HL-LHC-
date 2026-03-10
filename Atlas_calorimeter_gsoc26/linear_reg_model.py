# Importing essential libraries
import torch
import glob
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# used AI assistance for loading the data from the pytorch shards
# loading the train.pt files
shard_train_files = sorted(glob.glob('Atlas_calorimeter_gsoc26/train/*.pt'))

all_samples_lo = [] 
all_ene_lo = [] 

for shard_file in shard_train_files:
    shard = torch.load(shard_file,weights_only=True)
    all_samples_lo.append(shard['X'][:,1,:])
    all_ene_lo.append(shard['y'])

# concatenating all shards into one tensor
sample_lo_train = torch.cat(all_samples_lo)
ene_lo_train = torch.cat(all_ene_lo)
    
# loading the test.pt files
shard_files_test = sorted(glob.glob('Atlas_calorimeter_gsoc26/test/*.pt'))

all_samples_lo_test = [] 
all_ene_lo_test = [] 

for shard_file in shard_files_test:
    shard = torch.load(shard_file,weights_only=True)
    all_samples_lo_test.append(shard['X'][:,1,:])
    all_ene_lo_test.append(shard['y'])

# combining all test shards
sample_lo_test = torch.cat(all_samples_lo_test)
ene_lo_test = torch.cat(all_ene_lo_test)

# converting tensors to numpy arrays so they can be used with sklearn
X_train = sample_lo_train.numpy()
Y_train = ene_lo_train[:,0].numpy()

X_test = sample_lo_test.numpy()
Y_test = ene_lo_test[:,0].numpy()

# filtering out low energy values from training set since they are likely noise
mask_train = np.abs(Y_train) > 1
X_train = X_train[mask_train]
Y_train = Y_train[mask_train]

# creating and training a simple linear regression model
Model = LinearRegression()
Model.fit(X_train,Y_train)

# predicting energies for the test dataset
Y_pred_test = Model.predict(X_test)

# filtering test samples to remove very small or noisy energy values (0.6 sampled after testing multiple thresholds to find a good balance between noise reduction and retaining enough samples for analysis)
mask = np.abs(Y_test) > 1

Y_pred_filtered = Y_pred_test[mask]
Y_test_filtered = Y_test[mask]


# calculating residuals between predicted and true energies
residual = (Y_pred_filtered - Y_test_filtered) / Y_test_filtered

# computing statistics of the residual distribution
mean = np.mean(residual)
rms = np.sqrt(np.mean(residual**2))
intercept = Model.intercept_
coefficients = Model.coef_

print(f"Mean: {mean:.6f}")
print(f"RMS:  {rms:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"Coefficients: {coefficients}")
#shows how many samples passed the filter to give context to the residual statistics
print(f"Samples passing filter: {mask.sum()} out of {len(Y_test)}")

# plotting histogram of residual distribution
plt.figure(figsize=(12,10))
plt.hist(residual , bins = 100 , color='blue' , edgecolor = 'black')
plt.xlabel('(predicted - true) / true')
plt.ylabel('Count')
plt.title('Residual Distribution')
plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.6f}')
plt.legend()
plt.savefig('residual_histogram.png')
plt.show()


# creating a 2D histogram of residual vs true energy
# AI assistance was used for constructing this visualization
plt.figure(figsize=(12, 10))
plt.hist2d(Y_test_filtered, residual, bins=100, cmap='viridis')

plt.colorbar(label='Count')
plt.xlabel('True Energy')
plt.ylabel('(predicted - true) / true')
plt.title('Residual vs True Energy')

plt.savefig('residual_vs_energy.png')
plt.show()


#The bonus problem
# getting time values from the second column of the energy tensor for both train and test datasets
Y_time_train = ene_lo_train[:, 1].numpy()
Y_time_test  = ene_lo_test[:, 1].numpy()

# training a linear regression model to predict time values 
time_model = LinearRegression()
time_model.fit(sample_lo_train.numpy(), Y_time_train) 

# predicting time values for the test dataset using the trained model
Y_time_pred = time_model.predict(X_test)

# computing statistics of the residual distribution for time predictions
time_residual = Y_time_pred - Y_time_test  
time_mean = np.mean(time_residual)
time_rms  = np.sqrt(np.mean(time_residual**2))

print(f"Time Mean: {time_mean:.6f}")
print(f"Time RMS:  {time_rms:.6f}")
print(f"Time Intercept: {time_model.intercept_:.6f}")
print(f"Time Coefficients: {time_model.coef_}")
#ploting histogram of time residual distribution similar to the energy residual histogram since no clear format was specified for the bonus probelem 
plt.figure(figsize=(12,10))
plt.hist(time_residual, bins=100, color='green', edgecolor='black')
plt.xlabel('predicted time - true time')
plt.ylabel('Count')
plt.title('Time Residual Distribution')
plt.axvline(time_mean, color='red', linestyle='--', label=f'Mean: {time_mean:.4f}')
plt.legend()
plt.savefig('time_residual_histogram.png')
plt.show()