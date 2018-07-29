import numpy as np
import pandas as pd
import sys

__author__ = 'andreykoloskov'

def main():
    df = pd.read_csv(sys.argv[1])
    ap = pd.read_csv(sys.argv[2])

    #1.1
    cancel = df.groupby('CancellationCode').size().reset_index()
    max_cancel = np.array(cancel.sort_values(by=0).tail(1)['CancellationCode'])[0]
    print("max_flight_cancellation = ", max_cancel, "\n")

    #1.2
    dist = df.Distance
    avg_dist = dist.mean()
    min_dist = dist.min()
    max_dist = dist.max()
    print("avg = ", avg_dist)
    print("min = ", min_dist)
    print("max = ", max_dist)
    print("\n")

    #1.3
    min_dist_arr = df[df.Distance == min_dist]
    print("min_distance:")
    print(min_dist_arr[['Year', 'Month', 'DayofMonth', 'FlightNum', 'Distance']])
    other_dist = df[df.FlightNum.isin(min_dist_arr['FlightNum'])]
    print("all_distance")
    print(other_dist.Distance.unique())
    print('\n')

    #1.4
    orig = df.groupby('Origin').size().reset_index()
    airoport = np.array(orig.sort_values(by=0).tail(1)['Origin'])[0]
    city = np.array(ap[ap.iata == airoport]['city'])[0]
    print("max_flight_airoport = ", airoport)
    print("city = ", city, "\n")

    #1.5
    tm = df.groupby('Origin')['AirTime'].mean().reset_index()
    airoport = np.array(tm.sort_values(by='AirTime').tail(1)['Origin'])[0]
    city = np.array(ap[ap.iata == airoport]['city'])[0]
    print("max_avg_time_airoport = ", airoport)
    print("city = ", city, "\n")

    #1.6
    grp = df[df.DepDelay > 0][['Origin', 'DepDelay']]
    delay = grp.groupby('Origin').size().reset_index()
    airoport1 = np.array(delay.sort_values(by = 0)['Origin'])

    flt = df.groupby('Origin').count().reset_index()
    airoport2 = np.array(flt[flt.FlightNum >= 1000][['Origin']].reset_index()['Origin'])
    arr = [x for x in airoport1 if x in airoport2]
    airoport = arr[-1]
    print("max_delay_big_airoport = ", airoport)

if __name__ == '__main__':
    main()
