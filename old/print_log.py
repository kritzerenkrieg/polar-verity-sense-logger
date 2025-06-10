import asyncio
import signal
import csv
from bleak import BleakScanner, BleakClient
from datetime import datetime

# Global list to store log entries
log_entries = []

async def scan_for_devices():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    return devices

async def notification_handler(sender, data):
    """Handle incoming notifications from the BLE device."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {"timestamp": timestamp, "sender": sender, "data": data}
    log_entries.append(log_entry)
    print(f"Received notification from {sender}: {data}")

async def listen_to_characteristic(client, characteristic):
    """Start listening to a specific characteristic."""
    try:
        await client.start_notify(characteristic.uuid, notification_handler)
        print(f"Listening for notifications on {characteristic.uuid}...")

        # Keep listening as long as the client is connected
        while client.is_connected:
            await asyncio.sleep(1)  # Keep the task alive

    except Exception as e:
        print(f"Error with characteristic {characteristic.uuid}: {e}")

async def pair_with_device(device_address):
    try:
        async with BleakClient(device_address) as client:
            print(f"Connected to {client.address}")

            # Discover services and characteristics
            services = await client.get_services()
            characteristics = [char for service in services for char in service.characteristics]
            
            # Filter characteristics that support notifications or indications
            notifiable_characteristics = [
                char for char in characteristics if 'notify' in char.properties or 'indicate' in char.properties
            ]

            if not notifiable_characteristics:
                print("No characteristics support notifications or indications.")
                return

            # Start listening to all notifiable characteristics concurrently
            print("Setting up notifications for all available characteristics...")
            tasks = [listen_to_characteristic(client, char) for char in notifiable_characteristics]

            # Run all tasks in parallel
            await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Failed to connect to {device_address}: {e}")

def save_log_to_csv():
    """Save the log entries to a CSV file."""
    if log_entries:
        filename = f"ble_notifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'sender', 'data']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in log_entries:
                writer.writerow(entry)
        print(f"\nLog saved to {filename}.")
    else:
        print("\nNo notifications were received, so no log was saved.")

def signal_handler(sig, frame):
    """Handle the termination signal (Ctrl+C) to save log and exit."""
    print("\nTermination signal received. Saving log...")
    save_log_to_csv()
    asyncio.get_event_loop().stop()

async def main():
    devices = await scan_for_devices()
    
    # Display available devices
    for i, device in enumerate(devices):
        print(f"{i + 1}: {device.name or 'Unknown'} - {device.address}")

    # Select device
    choice = int(input("Select a device to pair with (number): ")) - 1
    if 0 <= choice < len(devices):
        device_address = devices[choice].address
        print(f"You selected {devices[choice].name or 'Unknown'} - {device_address}")
        
        await pair_with_device(device_address)
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Run the asyncio event loop
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
