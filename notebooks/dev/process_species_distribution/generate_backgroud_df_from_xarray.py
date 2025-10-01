import xarray as xr
import numpy as np

# First create the original dataset (as before)
x = [3,5,7]
y = [1,2,3]
temp_data = np.random.normal(15, 5, size=(3, 3))
temp_data2 = np.random.normal(20, 15, size=(3, 3))
ds = xr.Dataset(
    data_vars={
        'temperature': (('y', 'x'), temp_data),
        'temperature2': (('y', 'x'), temp_data2)
    },
    coords={
        'x': x,
        'y': y
    },
    attrs={
        'description': 'Synthetic temperature data'
    }
)


# Define the coordinates to mask
x_list = [3.5, 3]
y_list = [1, 2.5]

# Create a boolean mask (True where we want to keep values)
mask = ds.assign(mask=lambda x: (x.temperature * 0 + 1).astype(bool))
mask.mask.loc[dict(x=x_list, y=y_list)] = False

# Create new dataset with masked values
ds_masked = ds.copy()
ds_masked = ds.where(mask.mask)

print("Original dataset:")
print(ds)
print("Masked dataset:")
print(ds_masked)

# To verify the masking:
print("Values:")  
print(ds_masked.to_dataframe())
print(ds_masked.to_dataframe().dropna())    