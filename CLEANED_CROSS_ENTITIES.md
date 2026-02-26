# Cross-Entity Cleanup Report

## ✅ Completed Changes

### 1. SOURCE_NAME → PICKUP_POINT Migration
**Removed from SOURCE_NAME** (area-level locations that belong in PICKUP_POINT):
- Electronic City, Silk Board Junction, Majestic Bangalore
- Yeshwanthpur, BTM Layout, Marathahalli, Whitefield
- Kengeri, Banashankari, KR Puram, Hebbagodi, Attibele, Bommasandra
- Miyapur, Ameerpet, Kukatpally, LB Nagar, Uppal, Dilsukhnagar, Gachibowli
- Shamshabad, Kothapet, JNTU Hyderabad, Hitech City, Madhapur
- Koyambedu Omni Bus Stand, Guindy, T. Nagar, Velachery, Sholinganallur
- Siruseri, Poonamallee, Red Hills
- Vashi, Nerul, Borivali West, Andheri, Kurla
- Vyttila Mobility Hub
- Delhi Airport
- Hebbal
- Kalyan
- Ambala

**Kept in SOURCE_NAME** (city-level locations):
- Tambaram, Perungalathur, Chengalpattu (cities near Chennai)
- Panvel, Dombivli, Bhiwandi (cities near Mumbai)
- All other city names

### 2. DESTINATION_NAME → DROP_POINT Migration
**Removed from DESTINATION_NAME**:
- Electronic City
- Guindy
- Edappally

**Kept in DESTINATION_NAME** (city-level):
- All major cities and towns

### 3. Locations that can be BOTH (kept in both entities)
These are legitimately both city destinations AND drop/pickup points:
- Dadar (city area + station/drop point)
- Basti (town + drop point)
- Tuljapur (town + drop point)

## 📊 Impact Summary

| Entity Pair | Before | After | Removed |
|------------|--------|-------|---------|
| SOURCE_NAME ↔ PICKUP_POINT | ~40 overlaps | ~3 overlaps | ~37 area-level locations |
| DESTINATION_NAME ↔ DROP_POINT | ~5 overlaps | ~3 overlaps | 3 area-level locations |

## ✨ Expected Benefits

1. **Clearer Entity Boundaries**: Model will learn that:
   - SOURCE_NAME/DESTINATION_NAME = City-level locations
   - PICKUP_POINT/DROP_POINT = Area/landmark-level locations within cities

2. **Better Context Learning**: Model can now rely on:
   - "from Electronic City" → PICKUP_POINT (area within Bangalore)
   - "from Bangalore" → SOURCE_NAME (city)

3. **Reduced Training Confusion**: No more conflicting labels for the same location string

## 🔍 What Remains to Fix (for next step)

1. **BUS_TYPE ↔ SEAT_TYPE overlaps**:
   - Remove: `slepper`, `sleepr`, `slerper`, `seatr`, `seeter`, `seter` from BUS_TYPE

2. **BUS_TYPE ↔ BUS_FEATURES overlaps**:
   - Remove: `PRIME BUS` from BUS_FEATURES (keep "Prime Bus" in BUS_TYPE only)

