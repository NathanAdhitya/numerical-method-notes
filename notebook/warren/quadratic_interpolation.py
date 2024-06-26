# Given data points
x1, f1 = 1, 0
x2, f2 = 4, 1.386294
x3, f3 = 6, 1.791759

# Calculate b1, b2, and b3
b1 = f1
b2 = (f2 - f1) / (x2 - x1)
b3 = ((f3 - f2) / (x3 - x2) - b2) / (x3 - x1)

# Print b1, b2, b3
print(f"b1 = {b1}")
print(f"b2 = {b2}")
print(f"b3 = {b3}")

# Calculate f(x) for x2 to verify
fx2 = b1 + b2 * (x2 - x1) + b3 * (x2 - x1) * (x2 - x2)
print(f"f(x2) = {fx2}")

# Define the quadratic interpolation function
def quadratic_interpolation(x):
    return b1 + b2 * (x - x1) + b3 * (x - x1) * (x - x2)

# Estimate ln(2)
x_estimate = 2
ln2_estimate = quadratic_interpolation(x_estimate)

print(f"Estimated ln(2) = {ln2_estimate}")
