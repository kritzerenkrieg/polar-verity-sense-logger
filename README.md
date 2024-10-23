#Polarity Sense BLE Logger, by Kentaro Mas'ud Mizoguchi

##Usage
1. pair.py - to scan and pair a bluetooth device, then interact and get services of BLE device
2. print_log.py - to log all transmission from the chosen BLE device, and saves log to csv
3. print_hrv.py - this script processes the print_log.py into a HRV-to-time graph and timestamp-heartrate csv.

![This is the graph preview from sample data](https://i.ibb.co.com/0D4Nxfv/Figure-1.png)

##this is sample of output data of print_hrv.py
timestamp,heart_rate
2024-10-23 11:20:40,92
2024-10-23 11:20:41,93
2024-10-23 11:20:42,94
