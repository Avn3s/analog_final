# Analog_Final

## Sensor Drift

### Identifying Sensor Drift
Here, all the training data is used since they are assumed to be consistent.

In this project, data will be referred to as consistent, if all the sensor inputs are relatively correct, i.e. if sensors a, b, c and d give certain values, sensor f's value can be predicted.

Consistency is not compromised by problems in machine health, since when a machine component  malfunctions, all sensors (assumed to be in good condition) give readings that are relatively consistent with each other. An outlier reading suggests sensor drift.

We neglect the possibility of multiple sensors failing simultaneously with consistent sensor drift.After all, it is highly improbable that such a problem would arise, since our final model is expected to run in a continuous feedback loop.

### Correcting Sensor Drift
The model is trained on consistent readings. Any inconsistent reading is flagged, and it is put through the model to gain the corrected, consistent data.

This corrected data (made to be consistent) is then sent through the second model to determine machine health.

The corresponding sensor error is also flagged and informed to the user.


## Machine Health
The data inputted to this model is all consistent thanks to the previous model. Now, a simple model determines machine health and informs that to the user too.