import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np


def create_no_leap_epw(fin, fout, start_date='2020-01-01', change_year_to=None):
    # Read the first 8 lines as header
    with open(fin, 'r') as f:
        header = [next(f) for _ in range(8)]

    # Read the rest as data
    epw = pd.read_csv(fin, skiprows=8, header=None)

    # Generate corresponding datetime index
    dates = pd.date_range(start_date, periods=8784, freq='h')

    # Mask out Feb 29
    mask = ~((dates.month == 2) & (dates.day == 29))
    epw_clean = epw[mask]

    if change_year_to is not None:
        epw_clean.iloc[:, 0] = change_year_to

    # Write back to new file with header
    with open(fout, 'w') as f:
        f.writelines(header)
        epw_clean.to_csv(f, index=False, header=False)


def dew_point_formula(temp_c, rh, a, b):
    alpha = np.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


def calculate_dew_point(temp_c, rh):
    # Step 1: First estimate with water constants
    a1, b1 = 17.62, 243.12  # Water
    dew_point = dew_point_formula(temp_c, rh, a1, b1)

    # Step 2: If dew point < 0°C, recalculate with ice constants
    if dew_point < 0:
        a2, b2 = 22.46, 272.62  # Ice
        dew_point = dew_point_formula(temp_c, rh, a2, b2)

    return dew_point


def create_simplified_epw(fin, fout, no_sun=False, dry_air=False, no_rain=False, const_temp=None):
    # Read the header lines separately
    with open(fin, 'r') as f:
        header_lines = [next(f) for _ in range(8)]

    # Read the rest as data
    data = pd.read_csv(fin, skiprows=8, header=None)

    if no_sun:
        # zero out everything with light and irradiation
        # Zero out Extraterrestrial Horizontal Radiation, Extraterrestrial Direct Normal Radiation, Horizontal Infrared Radiation Intensity
        data.iloc[:, 10:13] = 0

        # Zero out solar radiation columns (GHI, DNI, DHI, IR radiation)
        data.iloc[:, 13:17] = 0

        # Zero out Direct Normal Illuminance, Diffuse Horizontal Illuminance, Zenith Luminance
        data.iloc[:, 17:20] = 0

    # make climate completely waterless
    if dry_air:
        # relative humidity and dew point
        data.iloc[:, 8] = 1
        data.iloc[:, 7] = data.apply(lambda row: calculate_dew_point(row.iloc[6], row.iloc[8]), axis=1)

    if no_rain:
        data.iloc[:, 26] = 9 # no weather observation (no precipitation => ignore col 27)
        data.iloc[:, 28] = 0
        data.iloc[:, 33] = 0
        data.iloc[:, 34] = 0

    # set constant temperature
    if const_temp is not None:
        data.iloc[:, 6] = const_temp

    # Write it back to a new EPW file
    with open(fout, 'w') as f:
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
    # filename_ida = '20250521_Results.csv'
    # filename_ida = '20250603_Results.csv'
    # floor_key = 'Floor, Deg-C'
    # filename_ida = '20250604_Results.csv'
    floor_key = 'Floor - Crawl space, Deg-C'
    # filename_ida = '20250605_Results_no_sun.csv'
    # filename_ida = '20250624_Results_no_sun.csv'
    # filename_ida = '20250624_Results_no_sun_dry.csv'
    # filename_ida = '20250624_Results_no_leap_no_sun_t0.csv'
    filename_ida = '20250624_Results_no_leap_no_sun_dry_t0.csv'

    path = 'C:/Users/shinterseer/Desktop/GFBAE/IDA_ICE_Simulation_20250521/'
    df_ida = pd.read_csv(path + filename_ida)
    base_time = pd.Timestamp('2021')
    df_ida['datetime'] = df_ida['Time'].apply(lambda h: base_time + pd.Timedelta(hours=h))
    df_ida.index = df_ida['datetime']
    df_ida.drop(columns=['Time', 'datetime'], inplace=True)

    # filename_iso = 'SimulationResults_without_sun_20250605.csv'
    # filename_iso = 'SimulationResults_with_sun_20250623.csv'
    # filename_iso = 'SimulationResults_20250624_no_leap_no_sun_t0.csv'
    filename_iso = 'SimulationResults_20250624_no_leap_no_sun_dry_t0.csv'
    df_iso52k = pd.read_csv(filename_iso, index_col=0)
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
    # print(df_ida.columns)
    # print(df_iso52k.columns)

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
            # Format date ticks
            # a.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))  # e.g. 'Jan 01, 2024'
            # a.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # e.g. 'Jan 01'

    # Optional: auto-rotate and align
    fig.autofmt_xdate()  # OR: plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.show(block=True)


if __name__ == "__main__":
    file_in = '2020_Berlin.epw'
    file_out = '2021_Berlin_no_leap.epw'
    file_out2 = '2021_Berlin_no_leap_no_sun_t0.epw'
    path = 'C:/Users/shinterseer/Desktop/GFBAE/GFBAE.Simulation/Example/'
    # create_no_leap_epw(path + file_in, path + file_out, change_year_to=2021)
    create_simplified_epw(path + file_out, path + file_out2, dry_air=False, no_rain=False, const_temp=0)

    main()
