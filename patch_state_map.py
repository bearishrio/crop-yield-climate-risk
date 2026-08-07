# Quick patch for state_map
with open(r"C:\Users\rioth\OneDrive\Desktop\Test\aggregate_climate.py", "r") as f:
    content = f.read()

old = 'state_map = {"Maharashtra": "MAHARASHTRA", "Punjab": "PUNJAB", "Orissa": "ODISHA"}'
new = 'state_map = {"Maharashtra": "Maharashtra", "Punjab": "Punjab", "Orissa": "Odisha"}'

content = content.replace(old, new)

with open(r"C:\Users\rioth\OneDrive\Desktop\Test\aggregate_climate.py", "w") as f:
    f.write(content)

print("Patched state_map")