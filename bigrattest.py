import sensor_drift.drift as sd
import machine_health.health as mh

print("Healthy Training Data: ")
healthy = sd.load_data("sensor_drift/train_data.csv")
print(healthy)
model = sd.learn(healthy)

print("\n\nInconsistent Data in the Test dataset:")
test = sd.load_data("sensor_drift/test_data.csv")
report = sd.check(model, test)

print(report[report.is_unusual][["suspect_columns", "n_suspects"]])

print("\n\nCorrected Test dataset:")
corrected = sd.apply_corrections(test, report)

print(corrected)

print("\n\nSensor Health:")

sensor_health = sd.sensor_health_index(report)
for sensor in sensor_health:
    print(f"\t- {sensor}_reliability: {sensor_health[sensor]}")

"""
Now the corrected dataset will be used as input for the
second model, that determines machine health.
"""

mh_data = mh.load_data("machine_health/train.csv")
print(mh_data)
x = mh_data.drop("machine_health", axis=1)
y = mh_data["machine_health"]

mh_model = mh.learn(x, y)
y_pred = mh.check(corrected, mh_model)
print(f"\n\nMachine Health Index: {y_pred.round(4) * 100}%")
