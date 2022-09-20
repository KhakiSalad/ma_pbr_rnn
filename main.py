import torch
from sklearn.ensemble import RandomForestRegressor
import transformer
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import pandas as pd
import optuna
from networks import IMVTensorLSTM
from captum.attr import IntegratedGradients, DeepLift, GradientShap, FeatureAblation, LRP

import util as u
import config as c
from lstm import LSTM1


def train_and_validate_transformer(settings):
    enc_seq_len = settings["window_size"]
    output_sequence_length = 1
    window_size = enc_seq_len + output_sequence_length
    in_features_encoder_linear_layer = 2048
    in_features_decoder_linear_layer = 2048

    train, val = u.load_and_preprocess_train()
    input_variables = c.cols
    training_indices = transformer.utils.get_indices_entire_sequence(
        data=train,
        window_size=window_size,
        step_size=1)
    training_data = transformer.utils.TransformerDataset(
        data=torch.tensor(train[input_variables].values).float(),
        indices=training_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=settings["dec_seq_len"],
        target_seq_len=output_sequence_length
    )
    train_loader = DataLoader(training_data, batch_size=64)

    val_indices = transformer.utils.get_indices_entire_sequence(
        data=val,
        window_size=window_size,
        step_size=1)
    val_data = transformer.utils.TransformerDataset(
        data=torch.tensor(val[input_variables].values).float(),
        indices=val_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=settings["dec_seq_len"],
        target_seq_len=output_sequence_length
    )
    val_loader = DataLoader(val_data, batch_size=64)

    src_mask = transformer.utils.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len
    ).cuda()

    tgt_mask = transformer.utils.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=output_sequence_length
    ).cuda()

    model = transformer.transformer_timeseries.TimeSeriesTransformer(
        input_size=len(input_variables),
        dec_seq_len=settings["dec_seq_len"],
        batch_first=True,
        num_predicted_features=1,
        dim_val=settings["dim_val"],
        out_seq_len=output_sequence_length,
        n_encoder_layers=settings["n_encoder_layers"],
        n_decoder_layers=settings["n_decoder_layers"],
        n_heads=settings["n_heads"],
        dim_feedforward_encoder=in_features_encoder_linear_layer,
        dim_feedforward_decoder=in_features_decoder_linear_layer,
    )

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for i in range(c.max_epochs):
        for src, trg, trg_y in train_loader:
            src = src.cuda()
            trg = trg.cuda()
            trg_y = trg_y.cuda()
            optimizer.zero_grad()
            output = model(
                src=src,
                tgt=trg,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            output = output.squeeze(1).reshape(-1)
            loss = criterion(output, trg_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        mse_val = 0
        for src, trg, trg_y in val_loader:
            src = src.cuda()
            trg = trg.cuda()
            trg_y = trg_y.cuda()
            output = model(
                src=src,
                tgt=trg,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            output = output.squeeze(1).reshape(-1)
            mse_val += criterion(output, trg_y).item() * src.shape[0]
    return model, mse_val / len(val)


def objective_transformer(trial):
    dim_val = trial.suggest_categorical('dim_val', [32, 64, 128, 512, 1024])
    n_heads = trial.suggest_categorical('n_heads', [8, 16])
    n_decoder_layers = trial.suggest_int('n_decoder_layers', 2, 6)
    n_encoder_layers = trial.suggest_int('n_encoder_layers', 2, 6)
    dec_seq_len = trial.suggest_int('dec_seq_len', 20, 100)
    window_size = trial.suggest_categorical('window_size', [12, 24, 48, 96])

    settings = {
        'dim_val': dim_val,
        'n_heads': n_heads,
        'n_decoder_layers': n_decoder_layers,
        'n_encoder_layers': n_encoder_layers,
        'dec_seq_len': dec_seq_len,
        'window_size': window_size
    }
    return train_and_validate_transformer(settings)[1]


def test_transformer(settings):
    if c.train_model:
        model, _ = train_and_validate_transformer(settings)
        torch.save(model.state_dict(), 'models/transformer')
    else:
        model = transformer.transformer_timeseries.TimeSeriesTransformer(
            input_size=len(c.cols),
            dec_seq_len=settings["dec_seq_len"],
            batch_first=True,
            num_predicted_features=1,
            dim_val=settings["dim_val"],
            out_seq_len=1,
            n_encoder_layers=settings["n_encoder_layers"],
            n_decoder_layers=settings["n_decoder_layers"],
            n_heads=settings["n_heads"],
            dim_feedforward_encoder=2048,
            dim_feedforward_decoder=2048,
        )
        model.load_state_dict(torch.load('models/transformer'))
        model.cuda()
        model.eval()
    output_sequence_length = 1
    test = u.load_data(c.dataset_path_test)
    test = u.standard_scaler(test)
    plt.plot(test)
    plt.savefig("test_fig")

    test_indices = transformer.utils.get_indices_entire_sequence(
        data=test,
        window_size=settings["window_size"] + 1,
        step_size=1)

    test_data = transformer.utils.TransformerDataset(
        data=torch.tensor(test[c.cols].values).float(),
        indices=test_indices,
        enc_seq_len=settings["window_size"],
        dec_seq_len=settings["dec_seq_len"],
        target_seq_len=output_sequence_length
    )
    test_data = DataLoader(test_data, batch_size=64)

    src_mask = transformer.utils.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=settings["window_size"]
    ).cuda()

    tgt_mask = transformer.utils.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=output_sequence_length
    ).cuda()

    preds = []
    true = []
    attributions_dl = []
    attributions_fa = []
    attributions_gs = []
    ig = DeepLift(model)
    fa = FeatureAblation(model)
    gs = GradientShap(model)
    model.eval()
    for src, trg, trg_y in test_data:
        src = src.cuda()
        trg = trg.cuda()
        trg_y = trg_y.cuda()
        output = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        output = output.squeeze(1)
        preds.append(output.detach().cpu().numpy())
        true.append(trg_y.detach().cpu().numpy())
        attributions_dl.append(ig.attribute((src, trg))[0].detach().cpu().numpy())
        attributions_fa.append(fa.attribute((src, trg))[0].detach().cpu().numpy())
        baselines = (torch.randn(src.shape).cuda(), torch.randn(trg.shape).cuda())
        attributions_gs.append(gs.attribute((src, trg), baselines)[0].detach().cpu().numpy())

    attributions_dl = np.concatenate(attributions_dl)
    attributions_fa = np.concatenate(attributions_fa)
    attributions_gs = np.concatenate(attributions_gs)
    for attributions, prefix in [(attributions_dl, 'dl'), (attributions_fa, 'fa'), (attributions_gs, 'gs')]:
        mean_attrib = attributions.mean(0)
        std_attrib = attributions.std(0)
        for i, name in enumerate(c.cols):
            plt.figure()
            plt.plot(mean_attrib[:, i])
            plt.fill_between(range(len(mean_attrib[:, i])), mean_attrib[:, i] + std_attrib[:, i], mean_attrib[: ,i] - std_attrib[:, i], color='lightgray')
            plt.savefig(f'attributions/{prefix}_transformer_{name}')
            plt.close()

    preds = np.concatenate(preds).reshape(-1)
    true = np.concatenate(true)
    plt.figure(figsize=(10, 5))
    plt.plot(range(settings["window_size"]+1, len(test)), preds, label='prediction')
    plt.plot(test.iloc[:settings["window_size"],0].values, color='lightgray')
    plt.plot(range(settings["window_size"]+1, len(test)), true, label='true')
    plt.xlabel('time')
    plt.ylabel('standardized relative density')
    plt.legend()
    plt.savefig('pred_transformer')

    mae = mean_absolute_error(preds, true)
    mse = mean_squared_error(preds, true)

    return mae, mse


def train_and_validate_imvlstm(settings):
    train, val = u.load_and_preprocess_train()
    train = create_subsequences(settings["window_size"], train)
    val = create_subsequences(settings["window_size"], val)
    train_loader = DataLoader(train, batch_size=64, shuffle=False)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)

    input_dim = len(c.cols)
    n_units = settings["n_units"]

    model = IMVTensorLSTM(input_dim, 1, n_units).cuda()
    criterion = nn.MSELoss()
    lr = settings["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epochs = c.max_epochs
    min_val_loss = c.min_val_loss
    counter = 0
    for i in range(epochs):
        mse_train = 0
        model.train()
        for j, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            y_pred, _, _ = model(batch_x)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            mse_train += loss.item() * batch_x.shape[0]
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mse_val = 0
            val_steps = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output, _, _ = model(batch_x)
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += criterion(output, batch_y).item() * batch_x.shape[0]
                val_steps += 1
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            counter = 0
        else:
            counter += 1
        if counter == c.patience:
            return model, mean_squared_error(true, preds)
    return model, mean_squared_error(true, preds)


def objective_imvlstm(trial):
    lr = trial.suggest_float('lr', 0.001, 0.006)
    n_units = trial.suggest_categorical('n_units', [32, 64, 128, 256, 512, 1024])
    window_size = trial.suggest_categorical('window_size', [12, 24, 48])
    settings = {
        'lr': lr,
        'n_units': n_units,
        'window_size': window_size
    }
    _, score = train_and_validate_imvlstm(settings)
    return score


def test_imvlstm(settings):
    if c.train_model:
        model, _ = train_and_validate_imvlstm(settings)
        torch.save(model.state_dict(), 'models/imvlstm')
    else:
        model = IMVTensorLSTM(len(c.cols), 1, settings["n_units"])
        model.load_state_dict(torch.load('models/imvlstm'))
        model.cuda()
    test = u.load_data(c.dataset_path_test)
    test = u.standard_scaler(test)
    test = create_subsequences(settings["window_size"], test)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    preds = []
    true = []
    alphas = []
    betas = []
    attributions_dl = []
    attributions_fa = []
    attributions_gs = []
    dl = DeepLift(model)
    fa = FeatureAblation(model)
    gs = GradientShap(model)
    model.eval()
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.cuda()
        output, a, b = model(batch_x)
        output = output.squeeze(1)
        alphas.append(a.detach().cpu().numpy())
        betas.append(b.detach().cpu().numpy())
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.numpy())

        # Für Post-Hoc Erklärungen IMV-LSTM Ausgabe anpassen (alphas, betas nicht ausgeben)
        #attributions_dl.append(dl.attribute(batch_x, target=0).detach().cpu().numpy())
        #attributions_fa.append(fa.attribute(batch_x).detach().cpu().numpy())
        #baselines = torch.randn(batch_x.shape).cuda()
        #attributions_gs.append(gs.attribute(batch_x, baselines).detach().cpu().numpy())

    #attributions_dl = np.concatenate(attributions_dl)
    #attributions_fa = np.concatenate(attributions_fa)
    #attributions_gs = np.concatenate(attributions_gs)
    #for attributions, prefix in [(attributions_dl, 'dl'), (attributions_fa, 'fa'), (attributions_gs, 'gs')]:
    #    mean_attrib = attributions.mean(0)
    #    std_attrib = attributions.std(0)
    #    for i, name in enumerate(c.cols):
    #        plt.figure()
    #        plt.plot(mean_attrib[:,i])
    #        plt.fill_between(range(len(mean_attrib[:, i])), mean_attrib[:, i] + std_attrib[:, i], mean_attrib[: ,i] - std_attrib[:, i], color='lightgray')
    #        plt.savefig(f'attributions/{prefix}_imvlstm_{name}')
    #        plt.close()

    preds = np.concatenate(preds)
    true = np.concatenate(true)
    alphas = np.concatenate(alphas)
    betas = np.concatenate(betas)
    alphas = alphas.mean(axis=0)
    betas = betas.mean(axis=0)
    alphas = alphas[..., 0]
    betas = betas[..., 0]
    alphas = alphas.transpose(1, 0)

    plt.figure(figsize=(10, 5))
    plt.plot(preds, label='prediction')
    plt.plot(true, label='true')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('standardized relative density')
    plt.savefig("prediction_imvlstm")

    fig, ax1 = plt.subplots(figsize=(10, 10))
    im = ax1.imshow(alphas)
    for (j, i), value in np.ndenumerate(alphas):
        ax1.text(i, j, f'{value:.2f}', ha='center', va='center')
    ax1.set_title("Importance of features and timesteps")
    labels = [f't-{i}' for i in range(settings["window_size"])[::-1]]
    plt.sca(ax1)
    plt.xticks(ticks=range(settings["window_size"]), labels=labels)
    plt.yticks(ticks=range(len(c.cols)), labels=c.cols)
    plt.tight_layout()
    plt.savefig("temp_importance")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.title("Feature importance")
    plt.bar(range(len(c.cols)), betas)
    plt.xticks(ticks=range(len(c.cols)), labels=list(c.cols), rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance")
    plt.close()
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    print(mean_squared_error(true[:192], true[:192]))
    return mae, mse


def create_subsequences(window_size, data):
    X = np.zeros((len(data), window_size, len(c.cols)))
    for i, name in enumerate(c.cols):
        for j in range(window_size):
            X[:, j, i] = data[name].shift(window_size - j - 1).fillna(method='bfill')
    y = data['RelativeDensity'].shift(-c.forecast_skip).fillna(method='ffill').values
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    dataset = TensorDataset(X, y)
    return dataset


def iterative_single_step(best_params):
    if c.train_model:
        model, _ = train_and_validate_imvlstm(best_params)
        torch.save(model.state_dict(), 'models/imvlstm')
    else:
        model = IMVTensorLSTM(len(c.cols), 1, best_params["n_units"])
        model.load_state_dict(torch.load('models/imvlstm'))
        model.cuda()
    model.eval()
    test = u.load_data(c.dataset_path_test)
    test = u.standard_scaler(test)

    preds = np.zeros((len(test), 48))
    for i in range(len(preds)):
        preds[i, :12] = test['RelativeDensity'].shift(-i).fillna(method='ffill')[0:12]

    for i in range(12, 48):
        test_ds = create_subsequences(best_params["window_size"], test)
        test_ds[:][0][:,:,0] = torch.Tensor(preds[:,i-12:i])
        output, _, _ = model(test_ds[:][0].cuda())
        output = output.squeeze(1)
        preds[:, i] = output.detach().cpu().numpy()
    for i in range(0, len(test), 50):
        plt.close()
        plt.plot(range(i+12, i+48), preds[i][12:], label=f'predictions starting at {i}')
        plt.plot(test['RelativeDensity'].values[:i+12], color='lightgray')
        plt.plot(range(i+12, min(i+48, len(test))),test['RelativeDensity'].values[i+12:i+48], label='true')
        plt.plot(range(i+48, len(test)), test['RelativeDensity'].values[i+48:],  color='lightgray')
        plt.xlabel('time')
        plt.ylabel('standardized relative density')
        plt.legend()
        plt.savefig(f'multistep_imv_start_{i}')
        plt.show()


def train_and_validate_rf(settings):
    train, val = u.load_and_preprocess_train()
    train = create_subsequences(c.window_size, train)
    val = create_subsequences(c.window_size, val)
    train_loader = DataLoader(train, batch_size=64, shuffle=False)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)
    n_estimators = settings['n_estimators']
    max_depth = settings['max_depth']
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(train[:][0].reshape((len(train), -1)), train[:][1])
    y_pred = rf.predict(val[:][0].reshape((len(val), -1)))
    return rf, mean_absolute_error(y_pred, val[:][1])


def test_rf(settings):
    test = u.load_data(c.dataset_path_test)
    test = u.standard_scaler(test)
    test = create_subsequences(c.window_size, test)
    rf, _ = train_and_validate_rf(settings)
    y_pred = rf.predict(test[:][0].reshape((len(test), -1)))
    return mean_absolute_error(y_pred, test[:][1]), mean_squared_error(y_pred, test[:][1])


def objective_rf(trial):
    settings = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 20)
    }
    return train_and_validate_rf(settings)[1]


def run_rf_experiment():
    if c.hyperparameter_tuning:
        print({item: c.__dict__[item] for item in dir(c) if not item.startswith("__")})
        study = optuna.create_study(study_name='rf')
        study.optimize(objective_rf, n_trials=c.n_trials_default)
        best_params_rf = study.best_params
        print(best_params_rf)
    else:
        pass
    rf_scores = np.zeros((c.training_repeats, 2))
    for i in range(c.training_repeats):
        rf_score = test_rf(best_params_rf)
        rf_scores[i] = rf_score
        print(f'imv lstm scores: \nmae: {rf_score[0]}\nmse: {rf_score[1]}')
    print(rf_scores.mean(0), rf_scores.std(0))


def run_imv_experiment():
    if c.hyperparameter_tuning:
        print({item: c.__dict__[item] for item in dir(c) if not item.startswith("__")})
        study = optuna.create_study(study_name='imv_lstm_horizon_1')
        study.optimize(objective_imvlstm, n_trials=c.n_trials_imvlstm)
        best_params_imv = study.best_params
        print(best_params_imv)
    else:
        #best_params_imv = {'lr': 0.00535610249466438, 'n_units': 32, 'window_size': 12}
        best_params_imv = {'lr': 0.005, 'n_units': 512, 'window_size': 12}
    imv_scores = np.zeros((c.training_repeats, 2))
    for i in range(c.training_repeats):
        imv_score = test_imvlstm(best_params_imv)
        imv_scores[i] = imv_score
        print(f'imv lstm scores: \nmae: {imv_score[0]}\nmse: {imv_score[1]}')
    print(imv_scores.mean(0), imv_scores.std(0))


def run_transformer_experiment():
    if c.hyperparameter_tuning:
        study = optuna.create_study(study_name='transformer_horizon_1')
        study.optimize(objective_transformer, n_trials=c.n_trials_transformer)
        best_params_trans = study.best_params
        print(best_params_trans)
    else:
        best_params_trans = {'dim_val': 64, 'n_heads': 16, 'n_decoder_layers': 2, 'n_encoder_layers': 4,
                             'dec_seq_len': 31, 'window_size': 96}
    t_scores = np.zeros((c.training_repeats, 2))
    for i in range(c.training_repeats):
        transformer_score = test_transformer(best_params_trans)
        t_scores[i] = transformer_score
        print(f'transformer scores: \nmae: {transformer_score[0]}\nmse: {transformer_score[1]}')
    print(t_scores.mean(0), t_scores.std(0))


def train_and_validate_lstm_baseline(settings):
    train, val = u.load_and_preprocess_train()
    train = create_subsequences(settings['window_size'], train)
    val = create_subsequences(settings['window_size'], val)
    train_loader = DataLoader(train, batch_size=64, shuffle=False)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)

    lr = settings['lr']
    hidden_size = settings['hidden_size']
    num_layers = settings['num_layers']

    lstm = LSTM1(len(c.cols), hidden_size, num_layers)
    lstm.cuda()
    optimizer = optim.Adam(lstm.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(c.max_epochs):
        mse_train = 0
        for j, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            y_pred = lstm(batch_x)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            mse_train += loss.item()*batch_x.shape[0]
            optimizer.step()
        with torch.no_grad():
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output = lstm(batch_x)
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
        preds = np.concatenate(preds)
        true = np.concatenate(true)
    return lstm, mean_squared_error(true, preds)


def test_lstm_baseline(settings):
    test = u.load_data(c.dataset_path_test)
    test = u.standard_scaler(test)
    test = create_subsequences(c.window_size, test)
    test_loader = DataLoader(test, batch_size=64)
    lstm_baseline, _ = train_and_validate_lstm_baseline(settings)
    lstm_baseline.eval()
    with torch.no_grad():
        preds = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = lstm_baseline(batch_x)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    return mean_absolute_error(preds, true), mean_squared_error(preds, true)


def objective_lstm_baseline(trial):
    settings = {
        'lr' : trial.suggest_float('lr', 0.0001, 0.01),
        'hidden_size' : trial.suggest_int('hidden_size', 5, 500),
        'num_layers' : trial.suggest_int('num_layers_lstm', 1, 5),
        'window_size': trial.suggest_categorical('window_size', [12, 48, 96, 192])
    }
    return train_and_validate_lstm_baseline(settings)[1]


def run_lstm_baseline_experiment():
    if c.hyperparameter_tuning:
        print({item: c.__dict__[item] for item in dir(c) if not item.startswith("__")})
        study = optuna.create_study(study_name='lstm_baseline')
        study.optimize(objective_lstm_baseline, n_trials=c.n_trials_default)
        best_params_lstm_baseline = study.best_params
        print(best_params_lstm_baseline)
    else:
        pass
    lstm_baseline_scores = np.zeros((c.training_repeats, 2))
    for i in range(c.training_repeats):
        lstm_baseline_score = test_lstm_baseline(best_params_lstm_baseline)
        lstm_baseline_scores[i] = lstm_baseline_score
        print(f'imv lstm scores: \nmae: {lstm_baseline_score[0]}\nmse: {lstm_baseline_score[1]}')
    print(lstm_baseline_scores.mean(0), lstm_baseline_scores.std(0))



if __name__ == "__main__":
    run_lstm_baseline_experiment()
    run_rf_experiment()
    run_imv_experiment()
    iterative_single_step({'lr': 0.005, 'n_units': 512, 'window_size': 12})
    run_transformer_experiment()

