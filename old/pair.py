import asyncio
from bleak import BleakScanner, BleakClient

async def scan_for_devices():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    return devices

async def pair_with_device(device_address):
    try:
        async with BleakClient(device_address) as client:
            print(f"Connected to {client.address}")
            # You can interact with the device here if needed.
            # For example, reading services or characteristics.
            services = await client.get_services()
            print("Services:", services)
            # Example of reading a characteristic
            # value = await client.read_gatt_char(YOUR_CHARACTERISTIC_UUID)
            # print("Characteristic value:", value)

            await asyncio.sleep(1)  # Keep the connection alive for a bit.
    except Exception as e:
        print(f"Failed to connect to {device_address}: {e}")

async def main():
    devices = await scan_for_devices()
    for i, device in enumerate(devices):
        print(f"{i + 1}: {device.name} - {device.address}")

    choice = int(input("Select a device to pair with (number): ")) - 1
    if 0 <= choice < len(devices):
        device_address = devices[choice].address
        await pair_with_device(device_address)
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    asyncio.run(main())
