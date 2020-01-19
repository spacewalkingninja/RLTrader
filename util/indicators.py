import ta

def add_indicators(df):
    df['RSI'] = ta.momentum.rsi(df["Close"])
    df['MFI'] = ta.momentum.money_flow_index(
        df["High"], df["Low"], df["Close"], df["Volume BTC"])
    df['TSI'] = ta.momentum.tsi(df["Close"])
    df['UO'] = ta.momentum.uo(df["High"], df["Low"], df["Close"])
    df['AO'] = ta.momentum.ao(df["High"], df["Low"])

    df['MACD_diff'] = ta.trend.macd_diff(df["Close"])
    df['Vortex_pos'] = ta.trend.vortex_indicator_pos(
        df["High"], df["Low"], df["Close"])
    df['Vortex_neg'] = ta.trend.vortex_indicator_neg(
        df["High"], df["Low"], df["Close"])
    df['Vortex_diff'] = abs(
        df['Vortex_pos'] -
        df['Vortex_neg'])
    df['Trix'] = ta.trend.trix(df["Close"])
    df['Mass_index'] = ta.trend.mass_index(df["High"], df["Low"])
    df['CCI'] = ta.trend.cci(df["High"], df["Low"], df["Close"])
    df['DPO'] = ta.trend.dpo(df["Close"])
    df['KST'] = ta.trend.kst(df["Close"])
    df['KST_sig'] = ta.trend.kst_sig(df["Close"])
    df['KST_diff'] = (
        df['KST'] -
        df['KST_sig'])
    df['Aroon_up'] = ta.trend.aroon_up(df["Close"])
    df['Aroon_down'] = ta.trend.aroon_down(df["Close"])
    df['Aroon_ind'] = (
        df['Aroon_up'] -
        df['Aroon_down']
    )

    df['BBH'] = ta.volatility.bollinger_hband(df["Close"])
    df['BBL'] = ta.volatility.bollinger_lband(df["Close"])
    df['BBM'] = ta.volatility.bollinger_mavg(df["Close"])
    df['BBHI'] = ta.volatility.bollinger_hband_indicator(
        df["Close"])
    df['BBLI'] = ta.volatility.bollinger_lband_indicator(
        df["Close"])
    df['KCHI'] = ta.volatility.keltner_channel_hband_indicator(df["High"],
                                                    df["Low"],
                                                    df["Close"])
    df['KCLI'] = ta.volatility.keltner_channel_lband_indicator(df["High"],
                                                    df["Low"],
                                                    df["Close"])
    df['DCHI'] = ta.volatility.donchian_channel_hband_indicator(df["Close"])
    df['DCLI'] = ta.volatility.donchian_channel_lband_indicator(df["Close"])

    df['ADI'] = ta.volume.acc_dist_index(df["High"],
                                  df["Low"],
                                  df["Close"],
                                  df["Volume BTC"])
    df['OBV'] = ta.volume.on_balance_volume(df["Close"],
                                     df["Volume BTC"])
    df['CMF'] = ta.volume.chaikin_money_flow(df["High"],
                                      df["Low"],
                                      df["Close"],
                                      df["Volume BTC"])
    df['FI'] = ta.volume.force_index(df["Close"],
                              df["Volume BTC"])
    df['EM'] = ta.volume.ease_of_movement(df["High"],
                                   df["Low"],
                                   df["Close"],
                                   df["Volume BTC"])
    df['VPT'] = ta.volume.volume_price_trend(df["Close"],
                                      df["Volume BTC"])
    df['NVI'] = ta.volume.negative_volume_index(df["Close"],
                                         df["Volume BTC"])

    df['DR'] = ta.others.daily_return(df["Close"])
    df['DLR'] = ta.others.daily_log_return(df["Close"])

    df.fillna(method='bfill', inplace=True)

    return df

