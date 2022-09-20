# data
dataset_path= "C:\\Users\\nilsk\\Documents\\Studium\\masterarbeit\\expose_testing\\datasets\\Log1.csv"
dataset_path_test= "C:\\Users\\nilsk\\Documents\\Studium\\masterarbeit\\expose_testing\\datasets\\Log2.csv"

# training loop
max_epochs = 500
patience = 90
min_val_loss= 9999
hyperparameter_tuning = True
train_model = True
training_repeats = 5

# resampling and windowing
forecast_skip=1
resample_mins= 30
window_size= 10

# learning task
cols= ["RelativeDensity","pH", "Temperature_C", "LightIntensity", "TankVolume", "Growth_Rate", "CO2_Injectionsper10min"]
#cols = ["RelativeDensity", "LightIntensity", "TankVolume", "CO2_Injectionsper10min"]
train_bound= 0.6

n_trials_default = 10
n_trials_transformer = n_trials_default
n_trials_imvlstm = 30