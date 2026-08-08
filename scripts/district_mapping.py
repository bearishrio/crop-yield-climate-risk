# District name mapping: Census 2011 -> ICRISAT
CENSUS_TO_ICRISAT = {
    # Maharashtra
    "Ahmadnagar": "Ahmednagar",
    "Amravati": "Amarawati",
    "Garhchiroli": "Gadchiroli",
    "Gondiya": "Gondia",
    "Nashik": "Nasik",
    "Yavatmal": "Yeotmal",
    # Punjab
    "Bathinda": "Bhatinda",
    "Firozpur": "Ferozpur",
    "Muktsar": "Shri Mukatsar Sahib",
    "Rupnagar": "Roopnagar",
    "Sahibzada Ajit Singh Nagar": "S.B.S Nagar",
    "Shahid Bhagat Singh Nagar": "S.B.S Nagar",
    # Odisha
    "Anugul": "Angul",
    "Balangir": "Bolangir",
    "Baleshwar": "Balasore",
    "Bauda": "Boudh",
    "Debagarh": "Deogarh",
    "Kandhamal": "Phulbani(Kandhamal)",
    "Kendujhar": "Keonjhar",
    "Khordha": "Khurda",
    "Nabarangapur": "Nawarangpur",
    "Subarnapur": "Sonepur",
    "Mayurbhanj": "Mayurbhanja",
}

# Reverse mapping
ICRISAT_TO_CENSUS = {v: k for k, v in CENSUS_TO_ICRISAT.items()}

# ICRISAT districts that exist in our cleaned yield panel (74 districts)
ICRISAT_DISTRICTS = {
    "Maharashtra": [
        "Ahmednagar", "Amarawati", "Aurangabad", "Bhandara", "Bid", "Chandrapur",
        "Dhule", "Gadchiroli", "Gondia", "Jalgaon", "Kolhapur", "Latur", "Nagpur",
        "Nanded", "Nandurbar", "Nasik", "Osmanabad", "Parbhani", "Pune", "Raigarh",
        "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Yeotmal"
    ],
    "Punjab": [
        "Amritsar", "Bhatinda", "Faridkot", "Fatehgarh Sahib", "Ferozpur", "Gurdaspur",
        "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Patiala",
        "Roopnagar", "S.B.S Nagar", "Sangrur", "Shri Mukatsar Sahib"
    ],
    "Orissa": [
        "Angul", "Balasore", "Bargarh", "Bhadrak", "Bolangir", "Boudh", "Cuttack",
        "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghapur", "Jajapur",
        "Jharsuguda", "Kalahandi", "Kendrapara", "Keonjhar", "Khurda", "Koraput",
        "Malkangiri", "Mayurbhanja", "Nawarangpur", "Nayagarh", "Nuapada",
        "Phulbani(Kandhamal)", "Puri", "Rayagada", "Sambalpur", "Sonepur", "Sundargarh"
    ]
}

# All ICRISAT districts as flat list
ALL_ICRISAT_DISTRICTS = []
for lst in ICRISAT_DISTRICTS.values():
    ALL_ICRISAT_DISTRICTS.extend(lst)

print(f"Total ICRISAT districts: {len(ALL_ICRISAT_DISTRICTS)}")
print(f"Maharashtra: {len(ICRISAT_DISTRICTS['Maharashtra'])}")
print(f"Punjab: {len(ICRISAT_DISTRICTS['Punjab'])}")
print(f"Orissa: {len(ICRISAT_DISTRICTS['Orissa'])}")