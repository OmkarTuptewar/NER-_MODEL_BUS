"""
Failed Pattern Templates
========================
Templates for queries that the model currently fails on.
These will be used to generate targeted training data.
"""

FAILED_TEMPLATES = [
    # ========================================
    # PATTERN 1: No preposition before DATE TIME
    # ========================================
    "show me buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "show buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "find buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "get buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "search buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "need buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "looking for buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    
    # ========================================
    # PATTERN 2: With "on" before DATE TIME
    # ========================================
    "show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "find buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "get buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "need buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "looking for buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    
    # ========================================
    # PATTERN 3: OPERATOR + BUS_TYPE combinations
    # ========================================
    "book {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "show {OPERATOR} {BUS_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "find {OPERATOR} {BUS_TYPE} service from {SOURCE_NAME} to {DESTINATION_NAME}",
    "I want {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "need {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "get me {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "looking for {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "show me {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "can I book {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "is {OPERATOR} {BUS_TYPE} available from {SOURCE_NAME} to {DESTINATION_NAME}",
    
    # ========================================
    # PATTERN 4: Complex with TRAVELER
    # ========================================
    "Can {TRAVELER} book {SEAT_TYPE} on {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Is {OPERATOR} {BUS_TYPE} available for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "{TRAVELER} want to book {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "book {SEAT_TYPE} for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "can {TRAVELER} book {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "show {SEAT_TYPE} for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME}",
    
    # ========================================
    # PATTERN 5: Minimal structure
    # ========================================
    "buses {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "bus {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "{SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "{SOURCE_NAME} to {DESTINATION_NAME} buses {DEPARTURE_DATE} {DEPARTURE_TIME}",
    
    # ========================================
    # PATTERN 6: DATE TIME at end without verbs
    # ========================================
    "buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "bus from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "I need bus from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "want bus from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    
    # ========================================
    # PATTERN 7: OPERATOR + BUS_TYPE + DATE TIME
    # ========================================
    "book {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "show {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "find {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "need {OPERATOR} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",

    # MISSING: SEMANTIC at the end of query
"I am looking for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, provide {SEMANTIC} options"
"looking for a bus from {SOURCE_NAME} to {DESTINATION_NAME}, please provide {SEMANTIC} options"
"show me a bus from {SOURCE_NAME} to {DESTINATION_NAME}, I need {SEMANTIC} option"

# MISSING: "Please provide" with SEMANTIC
"Please provide me with {SEMANTIC} options from {SOURCE_NAME} to {DESTINATION_NAME}"
"please provide {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}"
"provide me with the {SEMANTIC} options from {SOURCE_NAME} to {DESTINATION_NAME}"

# MISSING: Two-sentence pattern
"I am looking for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}. Please provide {SEMANTIC} options"
"looking for bus from {SOURCE_NAME} to {DESTINATION_NAME}. Show me {SEMANTIC} option"


# Round trip with proper ARRIVAL_DATE
"book {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with return on {ARRIVAL_DATE}",
"book {SEAT_TYPE} tickets from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} returning on {ARRIVAL_DATE}",
"two {SEAT_TYPE} tickets for round trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with return on {ARRIVAL_DATE}",
"confirm {SEAT_TYPE} tickets for round trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with return on {ARRIVAL_DATE}",
"Please confirm {SEAT_TYPE} tickets from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} returning on {ARRIVAL_DATE}",
"book round trip from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} returning {ARRIVAL_DATE}",

     "I want to apply {COUPON_CODE} for my journey with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}"
     "I want to apply the coupon code {COUPON_CODE} for my journey with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}"
     "apply {COUPON_CODE} for journey with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}"
     "I want to use {COUPON_CODE} for my journey with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}"

]
