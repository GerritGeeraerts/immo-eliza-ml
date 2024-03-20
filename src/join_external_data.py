from utils import load_data, join_data

print("make sure to download the REFNIS_CODES.geojson file from the link in the README.md file!")

df = load_data()
df = join_data(df)  # save the joined data to the intermediate folder
