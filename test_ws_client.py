import asyncio
import json
import websockets

async def test_client():
    uri = "ws://localhost:8765"
    print(f"Menghubungkan ke {uri} ...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Berhasil terhubung ke WebSocket Server!\n")
            print("Mendengarkan data broadcast (akan menampilkan 6 pesan pertama):")
            print("="*60)
            
            for i in range(6):
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"[{i+1}] Type: {data.get('type')} | Status: {data.get('status')}")
                if data.get("type") == "fatigue":
                    print(f"    Composite Score: {data.get('composite_score')} | Data Quality: {data.get('data_quality')}")
                    print(f"    Rekomendasi: {data.get('recommendation')}")
                elif data.get("type") == "dry_eye":
                    print(f"    Incomplete Blink: {data.get('incomplete_blink_ratio')} | PERCLOS: {data.get('perclos')}")
                elif data.get("type") == "myopia_risk":
                    print(f"    Jarak: {data.get('distance_cm')} cm | Warning Jarak: {data.get('distance_warning')}")
                    print(f"    Break State: {data.get('break_state')} | Screen Time: {data.get('screen_time_minutes')} min")
                print("-" * 60)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_client())
