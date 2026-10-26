from sensor_drift.drift import *

print("Healthy Training Data: ")
healthy = load_data("sensor_drift/train_data.csv")
print(healthy)
model = learn(healthy)

print("\n\nInconsistent Data in the Test dataset:")
test = load_data("sensor_drift/test_data.csv")
report = check(model, test)

print(report[report.is_unusual][["suspect_columns", "n_suspects"]])

print("\n\nCorrected Test dataset:")
corrected = apply_corrections(test, report)

print(corrected)

print("\n\nSensor Health:")

sensor_health = sensor_health_index(report)
for sensor in sensor_health:
    print(f"\t- {sensor}_reliability: {sensor_health[sensor]}")
