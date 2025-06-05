import pandas as pd
import matplotlib.pyplot as plt


def check_epw():
    from pvlib.iotools import epw
    path = 'C:/Users/shinterseer/Desktop/GFBAE/GFBAE.Simulation/Example/'
    file_out = '2020_Berlin_no_sun.epw'

    epw_data = epw.read_epw(path + file_out)

    epw_df = epw_data[0]
    weather_data = epw_df.loc[
                   :,
                   [
                       "temp_air",  # Air temperature
                       "ghi",  # Global Horizontal Radiation
                       "dhi",  # Diffuse Horizontal Radiation
                   ]
                   ]

def create_zero_sun_berlin():
    # Read the header lines separately
    path = 'C:/Users/shinterseer/Desktop/GFBAE/GFBAE.Simulation/Example/'
    file_in = '2020_Berlin.epw'
    file_out = '2020_Berlin_no_sun.epw'
    with open(path + file_in, 'r') as f:
        header_lines = [next(f) for _ in range(8)]

    # Read the rest as data
    data = pd.read_csv(path + file_in, skiprows=8, header=None)

    # zero out everything with light and irradiation
    # Zero out Extraterrestrial Horizontal Radiation, Extraterrestrial Direct Normal Radiation, Horizontal Infrared Radiation Intensity
    data.iloc[:, 10:13] = 0

    # Zero out solar radiation columns (GHI, DNI, DHI, IR radiation)
    data.iloc[:, 13:17] = 0

    # Zero out Direct Normal Illuminance, Diffuse Horizontal Illuminance, Zenith Luminance
    data.iloc[:, 17:20] = 0


    # Write it back to a new EPW file
    with open(path + file_out, 'w') as f:
        f.writelines(header_lines)
        data.to_csv(f, index=False, header=False)


def get_outside_temperature(filename='outside_temperature_unsampled.csv',
                            path='C:/Users/shinterseer/Desktop/GFBAE/IDA_ICE_Simulation_20250521/',
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
    # df_ida_ice_outside = get_outside_temperature()
    pass
    # filename_ida = '20250603_Results.csv'
    # floor_key = 'Floor, Deg-C'
    # filename_ida = '20250604_Results.csv'
    floor_key = 'Floor - Crawl space, Deg-C'
    filename_ida = '20250605_Results_no_sun.csv'


    path = 'C:/Users/shinterseer/Desktop/GFBAE/IDA_ICE_Simulation_20250521/'
    df_ida = pd.read_csv(path + filename_ida)
    base_time = pd.Timestamp('2020')
    df_ida['datetime'] = df_ida['Time'].apply(lambda h: base_time + pd.Timedelta(hours=h))
    df_ida.index = df_ida['datetime']
    df_ida.drop(columns=['Time', 'datetime'], inplace=True)

    # df = pd.read_csv(filename, index_col=0)
    # df.index = pd.to_datetime(df.index)
    # df_iso52k = pd.read_csv('SimulationResults_with_sun_20250602.csv', index_col=0)
    df_iso52k = pd.read_csv('SimulationResults_without_sun_20250605.csv', index_col=0)
    df_iso52k.index = pd.to_datetime(df_iso52k.index)

    # Step 2: Combine row 0 with original column names
    new_columns = [
        f"{col.strip()} {str(df_iso52k.iloc[0, idx]).strip()}"
        for idx, col in enumerate(df_iso52k.iloc[0].index)
    ]

    # Step 3: Apply new column names and drop the first row
    df_iso52k.columns = new_columns
    # df_iso52k.reset_index(drop=True, inplace=True)
    df_iso52k = df_iso52k.iloc[1:, :]
    df_iso52k = df_iso52k.apply(pd.to_numeric, errors='coerce')
    print(df_ida.columns)
    print(df_iso52k.columns)

    x = 0
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(df_ida["TAIR, Deg-C"], label="external temperature IDA ICE", color='grey')
    axs[0, 0].plot(df_iso52k['Unnamed: 1 theta_e_a'], label='external temperature iso52k', color='grey')

    axs[0, 0].plot(df_ida["Ceiling - Roof, Deg-C"], label="Roof_IDA ICE")
    axs[0, 0].plot(df_iso52k['Flat Roof: pli 4 (INT) theta_int_a'], label='roof iso52k', linestyle="dashed")
    axs[0, 0].plot(df_ida.index, df_ida['Mean air temperature, Deg-C'], label="Air temperature  IDA ICE", linewidth=0.5, color='black')
    axs[0, 0].plot(df_iso52k.index, df_iso52k['Zone "Z1".2 theta_int_a'], label='Air temperature iso52k', linewidth=0.5, color='black', linestyle="dashed")

    axs[0, 1].plot(df_ida[floor_key], label="Floor_IDA ICE")
    axs[0, 1].plot(df_iso52k['Ground Floor: pli 4 (INT) theta_int_a'], label='Floor iso52k', linestyle="dashed")
    axs[0, 1].plot(df_ida.index, df_ida['Mean air temperature, Deg-C'], label="Air temperature  IDA ICE", linewidth=0.5, color='black')
    axs[0, 1].plot(df_iso52k.index, df_iso52k['Zone "Z1".2 theta_int_a'], label='Air temperature iso52k', linewidth=0.5, color='black', linestyle="dashed")

    axs[1, 0].plot(df_ida['Wall 1 - f1, Deg-C'], label='Wall N - IDA ICA')
    axs[1, 0].plot(df_iso52k['Wall N: pli 4 (INT) theta_int_a'], label='Wall N iso52k', linestyle="dashed")
    axs[1, 0].plot(df_ida['Wall 3 - f3, Deg-C'], label='Wall S - IDA ICA')
    axs[1, 0].plot(df_iso52k['Wall S: pli 4 (INT) theta_int_a'], label='Wall S iso52k', linestyle="dashed")

    axs[1, 1].plot(df_ida["TAIR, Deg-C"], label="external temperature IDA ICE", color='grey')
    axs[1, 1].plot(df_iso52k['Unnamed: 1 theta_e_a'], label='external temperature iso52k', color='grey')
    axs[1, 1].plot(df_ida['Operative temperature, Deg-C'], label='Operative temperature IDA ICE')
    axs[1, 1].plot(df_iso52k['Zone "Z1".3 theta_int_op'], label='operative temperature iso52k', linestyle='dashed')

    for ax in axs:
        for a in ax:
            a.grid(True)
            a.legend()
    plt.show(block=True)


if __name__ == "__main__":
    # create_zero_sun_berlin()
    # check_epw()
    main()
