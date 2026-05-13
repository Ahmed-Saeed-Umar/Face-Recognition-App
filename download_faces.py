import requests, os, time

os.makedirs("dataset/NOT_ME", exist_ok=True)

for i in range(100):
    response = requests.get("https://thispersondoesnotexist.com", headers={"User-Agent": "Mozilla/5.0"})
    with open(f"dataset/NOT_ME/face_{i+1}.jpg", "wb") as f:
        f.write(response.content)
    print(f"Downloaded {i+1}/100")
    time.sleep(1)  

print("Done!")