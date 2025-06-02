import pandas as pd
import matplotlib.pyplot as plt


def get_outside_temperature(filename='outside_temperature_unsampled.csv',
                            path='C:/Users/shinterseer/Desktop/GFBAE/IDA_ICE_Simulation_20252105/',
                            plot=False,
                            start_timestamp='2020-01-01'):
    df = pd.read_csv(path + filename)
    hours_column = df.columns[0]
    df['hour_int'] = df[hours_column].round().astype(int)
    df = df[(df['hour_int'] >= 1) & (df['hour_int'] <= 8760)]
    df_resampled = df.groupby('hour_int').mean().reset_index()
    all_hours = pd.DataFrame({'hour_int': range(1, 8761)})
    df_resampled = all_hours.merge(df_resampled, on='hour_int', how='left')

    # Optional: fill missing values (NaNs) by interpolation or filling
    df_resampled = df_resampled.interpolate(method='linear')
    # or fill forward/backward
    df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
    # df_resampled = df_resampled.drop(columns=['hour_float'], errors='ignore')

    if plot:
        plt.plot(df_resampled['hour_int'], df_resampled[df_resampled.columns[-1]])
        plt.plot(df['Time'], df[df.columns[-2]])
        plt.show(block=True)

    base_time = pd.Timestamp(start_timestamp)
    df_resampled['datetime'] = df_resampled['hour_int'].apply(lambda h: base_time + pd.Timedelta(hours=h))

    # Set datetime as index
    df_resampled = df_resampled.set_index('datetime')
    df_resampled.drop(columns=['Time', 'hour_int'], inplace=True)
    return df_resampled


def main():
    # get_outside_temperature_mistral()
    df_ida_ice_outside = get_outside_temperature()
    pass
    filename = "2025.05.28_Test_01_Simple-house_results comparison.csv"
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    # df_TU = pd.read_csv('SimulationResults.csv', index_col=0)
    # df_TU.index = pd.to_datetime(df_TU.index)
    print(df.columns)
    x = 0
    fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(df.index, df["Roof_IDA ICE"], label="Roof_IDA ICE")
    # axs[0, 0].plot(df.index, df["Roof:pli 4 (INT)TU"], label="Roof:pli 4 (INT)", linestyle="dashed")
    # axs[0, 0].plot(df.index, df["Mean air temperature  [°C]IDA ICE"], label="Mean air temperature  [°C]IDA ICE", linewidth=0.5, color='black')
    # axs[0, 0].plot(df.index, df['Zone "Z1"_3_theta_int_aTU'], label='Zone "Z1"_3_theta_int_aTU', linewidth=0.5, color='black', linestyle="dashed")
    axs[0, 0].plot(df_ida_ice_outside["TAIR, Deg-C"], label="TAIR, Deg-C, IDA ICE")
    axs[0, 0].plot(df["external temperatue TU"], label="external temperatue TU")

    axs[0, 1].plot(df.index, df["Floor_IDA ICE"], label="Floor_IDA ICE")
    axs[0, 1].plot(df.index, df["Floor:pli 4 (INT)TU"], label="Floor:pli 4 (INT)TU", linestyle="dashed")
    axs[0, 1].plot(df.index, df["Mean air temperature  [°C]IDA ICE"], label="Mean air temperature  [°C]IDA ICE", linewidth=0.5, color='black')
    axs[0, 1].plot(df.index, df['Zone "Z1"_3_theta_int_aTU'], label='Zone "Z1"_3_theta_int_aTU', linewidth=0.5, color='black', linestyle="dashed")

    axs[1, 0].plot(df.index, df["Wall - f1_NIDA ICE"], label="Wall - f1_NIDA ICE")
    axs[1, 0].plot(df.index, df["Wall N: pli 4 (INT)TU"], label="Wall N: pli 4 (INT)TU", linestyle="dashed")
    axs[1, 0].plot(df.index, df["Wall-f3_SIDA ICE"], label="Wall-f3_SIDA ICE")
    axs[1, 0].plot(df.index, df["Wall S: pli 4 (INT)TU"], label="Wall S: pli 4 (INT)TU", linestyle="dashed")
    axs[1, 0].plot(df.index, df["Mean air temperature  [°C]IDA ICE"], label="Mean air temperature  [°C]IDA ICE", linewidth=0.5, color='black')
    axs[1, 0].plot(df.index, df['Zone "Z1"_3_theta_int_aTU'], label='Zone "Z1"_3_theta_int_aTU', linewidth=0.5, color='black', linestyle="dashed")

    axs[1, 1].plot(df.index, df["Mean air temperature  [°C]IDA ICE"], label="Mean air temperature  [°C]IDA ICE")
    axs[1, 1].plot(df.index, df['Zone "Z1"_3_theta_int_aTU'], label='Zone "Z1"_3_theta_int_aTU', linestyle="dashed")
    axs[1, 1].plot(df.index, df["Operative temperature [°C]IDA ICE"], label="Operative temperature [°C]IDA ICE")
    axs[1, 1].plot(df.index, df['Zone "Z1"_4_theta_int_opTU'], label='Zone "Z1"_4_theta_int_opTU', linestyle="dashed")

    for ax in axs:
        for a in ax:
            a.grid(True)
            a.legend()
    plt.show(block=True)


if __name__ == "__main__":
    main()
