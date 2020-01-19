import os
import gym
import optuna
import pandas as pd
import numpy as np

import cryptocompare
import datetime

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators

curr_idx = -1
reward_strategy = 'sortino'
input_data_file = os.path.join('data', 'coinbase_hourly.csv')
params_db_file = 'sqlite:///params.db'

study_name = 'ppo2_' + reward_strategy
study = optuna.load_study(study_name=study_name, storage=params_db_file)
params = study.best_trial.params
params = {'cliprange': 0.3838833398771404, 'confidence_interval': 0.8100538836579667, 'ent_coef': 2.2986983190512612e-06, 'forecast_len': 4.730444664178924, 'gamma': 0.9471575125156161, 'lam': 0.8983738705323411, 'learning_rate': 0.0015741188587479822, 'n_steps': 243.43124303268107, 'noptepochs': 35.778727741355404}
print("Training PPO2 agent with params:", params)
print("Best trial reward:", -1 * study.best_trial.value)


df = cryptocompare.get_historical_price_hour('BTC', curr='USD')
print(df)
df = pd.DataFrame(df['Data'])
df.columns = ['Close', 'High', 'Low', 'Open', 'Date', 'Volume BTC', 'Volume USD']
#df['Date'] = pd.to_datetime(df['Date'],unit='s')
df['Date'] = df['Date'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M:%S'))


df.set_index('Date')
#df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

df = df.sort_values(['Date'])
#df = pd.read_csv(input_data_file)
#df = df.drop(['Symbol'], axis=1)
#df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

print(df)
test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    train_df, reward_func=reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, reward_func=reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

if curr_idx == -1:
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
            tensorboard_log=os.path.join('.', 'tensorboard'), **model_params)
else:
    model = PPO2.load(os.path.join('.', 'agents', 'ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl'), env=train_env)

for idx in range(curr_idx + 1, 10):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)

    obs = test_env.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        reward_sum += reward

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save(os.path.join('.', 'agents', 'ppo2_' + reward_strategy + '_' + str(idx) + '.pkl'))
