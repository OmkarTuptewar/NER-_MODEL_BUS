TEMPLATES = [
    # Basic source-destination queries
    "bus from {SRC} to {DEST}",
    "buses from {SRC} to {DEST}",
    "{SRC} to {DEST} bus",
    "{SRC} to {DEST} buses",
    "show buses from {SRC} to {DEST}",
    "find bus from {SRC} to {DEST}",
    "search {SRC} to {DEST}",
    
    # With bus type
    "{BUS_TYPE} bus from {SRC} to {DEST}",
    "{BUS_TYPE} buses from {SRC} to {DEST}",
    "bus from {SRC} to {DEST} {BUS_TYPE}",
    "show {BUS_TYPE} buses from {SRC} to {DEST}",
    
    # With seat type
    "{SEAT_TYPE} bus from {SRC} to {DEST}",
    "{SEAT_TYPE} buses from {SRC} to {DEST}",
    "bus from {SRC} to {DEST} {SEAT_TYPE}",
    "{SRC} to {DEST} {SEAT_TYPE} bus",
    
    # With both bus and seat type
    "{BUS_TYPE} {SEAT_TYPE} bus from {SRC} to {DEST}",
    "{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST}",
    "{SEAT_TYPE} {BUS_TYPE} bus from {SRC} to {DEST}",
    "show {BUS_TYPE} {SEAT_TYPE} buses from {SRC} to {DEST}",
    
    # With time only
    "bus from {SRC} to {DEST} {TIME}",
    "{TIME} bus from {SRC} to {DEST}",
    "{SRC} to {DEST} {TIME}",
    "buses from {SRC} to {DEST} in the {TIME}",
    "{BUS_TYPE} bus from {SRC} to {DEST} {TIME}",
    
    # With date only
    "bus from {SRC} to {DEST} {DATE}",
    "{SRC} to {DEST} bus {DATE}",
    "buses from {SRC} to {DEST} for {DATE}",
    "{DATE} bus from {SRC} to {DEST}",
    "{BUS_TYPE} bus from {SRC} to {DEST} {DATE}",
    
    # With DATE and TIME separately (important for distinguishing)
    "bus from {SRC} to {DEST} on {DATE} at {TIME}",
    "bus from {SRC} to {DEST} {DATE} in the {TIME}",
    "{DATE} {TIME} bus from {SRC} to {DEST}",
    "buses from {SRC} to {DEST} {DATE} {TIME}",
    "{BUS_TYPE} bus from {SRC} to {DEST} {DATE} {TIME}",
    "{SEAT_TYPE} bus from {SRC} to {DEST} on {DATE} {TIME}",
    "show buses from {SRC} to {DEST} for {DATE} in {TIME}",
    "{SRC} to {DEST} {DATE} departing {TIME}",
    
    # With operator
    "{OPERATOR} bus from {SRC} to {DEST}",
    "{OPERATOR} buses from {SRC} to {DEST}",
    "show {OPERATOR} buses from {SRC} to {DEST}",
    "{OPERATOR} {BUS_TYPE} bus from {SRC} to {DEST}",
    "{OPERATOR} {SEAT_TYPE} from {SRC} to {DEST}",
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST}",
    
    # With boarding point only
    "bus from {SRC} to {DEST} pickup at {BOARDING_POINT}",
    "bus from {SRC} to {DEST} boarding from {BOARDING_POINT}",
    "{SRC} to {DEST} from {BOARDING_POINT}",
    "pickup {BOARDING_POINT} bus to {DEST}",
    "bus from {SRC} to {DEST} pickup {BOARDING_POINT}",
    "{BUS_TYPE} bus from {SRC} to {DEST} boarding {BOARDING_POINT}",
    
    # With dropping point only
    "bus from {SRC} to {DEST} drop at {DROPPING_POINT}",
    "bus from {SRC} to {DEST} dropping {DROPPING_POINT}",
    "{SRC} to {DEST} drop {DROPPING_POINT}",
    "bus from {SRC} to {DEST} drop {DROPPING_POINT}",
    "{BUS_TYPE} bus from {SRC} to {DEST} dropping at {DROPPING_POINT}",
    
    # With BOTH boarding and dropping points (important!)
    "bus from {SRC} to {DEST} pickup at {BOARDING_POINT} drop at {DROPPING_POINT}",
    "bus from {SRC} to {DEST} boarding {BOARDING_POINT} dropping {DROPPING_POINT}",
    "{SRC} to {DEST} from {BOARDING_POINT} to {DROPPING_POINT}",
    "bus from {SRC} to {DEST} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    "{BUS_TYPE} bus from {SRC} to {DEST} boarding at {BOARDING_POINT} dropping at {DROPPING_POINT}",
    "{SEAT_TYPE} from {SRC} to {DEST} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    "show buses from {SRC} to {DEST} boarding {BOARDING_POINT} drop {DROPPING_POINT}",
    
    # Complex queries with multiple entities
    "{BUS_TYPE} {SEAT_TYPE} bus from {SRC} to {DEST} {DATE}",
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST}",
    "show {OPERATOR} {BUS_TYPE} buses from {SRC} to {DEST} {DATE}",
    "{SEAT_TYPE} bus from {SRC} to {DEST} {TIME} pickup {BOARDING_POINT}",
    "{OPERATOR} {BUS_TYPE} from {SRC} to {DEST} {DATE} {TIME}",
    "{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME}",
    
    # FULL 9-ENTITY TEMPLATES (critical for comprehensive detection)
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME} pickup at {BOARDING_POINT} drop at {DROPPING_POINT}",
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} bus from {SRC} to {DEST} on {DATE} at {TIME} boarding {BOARDING_POINT} dropping {DROPPING_POINT}",
    "book {OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} for {DATE} {TIME} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    "{BUS_TYPE} {SEAT_TYPE} {OPERATOR} bus {SRC} to {DEST} {DATE} {TIME} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    "show {OPERATOR} {BUS_TYPE} {SEAT_TYPE} buses from {SRC} to {DEST} {DATE} in the {TIME} boarding from {BOARDING_POINT} dropping at {DROPPING_POINT}",
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} {SRC} to {DEST} {DATE} {TIME} from {BOARDING_POINT} to {DROPPING_POINT}",
    "find {OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} on {DATE} {TIME} boarding {BOARDING_POINT} drop {DROPPING_POINT}",
    "{OPERATOR} {SEAT_TYPE} {BUS_TYPE} bus from {SRC} to {DEST} {DATE} {TIME} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    
    # 7-8 entity combinations
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME}",
    "{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME} pickup {BOARDING_POINT}",
    "{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME} drop {DROPPING_POINT}",
    "{OPERATOR} {BUS_TYPE} from {SRC} to {DEST} {DATE} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    "{OPERATOR} {SEAT_TYPE} from {SRC} to {DEST} {TIME} boarding {BOARDING_POINT} dropping {DROPPING_POINT}",
    "{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    "{OPERATOR} bus from {SRC} to {DEST} {DATE} {TIME} pickup {BOARDING_POINT} drop {DROPPING_POINT}",
    
    # Natural language variations
    "I need a bus from {SRC} to {DEST}",
    "looking for {BUS_TYPE} bus from {SRC} to {DEST}",
    "want to travel from {SRC} to {DEST} {DATE}",
    "book bus from {SRC} to {DEST}",
    "book {OPERATOR} bus to {DEST}",
    "I want {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME}",
    "need {OPERATOR} bus from {SRC} to {DEST} {DATE}",
    
    # Natural language with "and" context (important: "and" should NOT be an entity)
    "book a bus from {SRC} to {DEST} and take a {SEAT_TYPE} bus",
    "bus from {SRC} to {DEST} for {DATE} and get a {SEAT_TYPE}",
    "I want to book from {SRC} to {DEST} and need {BUS_TYPE}",
    "travel from {SRC} to {DEST} {DATE} and prefer {BUS_TYPE} {SEAT_TYPE}",
    "find {OPERATOR} from {SRC} to {DEST} and take {SEAT_TYPE}",
    "book a bus from {SRC} to {DEST} for {DATE} and take a {SEAT_TYPE} bus",
    "show buses from {SRC} to {DEST} and filter by {BUS_TYPE}",
    
    # Conversational patterns
    "please book a bus from {SRC} to {DEST}",
    "can you find bus from {SRC} to {DEST} {DATE}",
    "help me book {BUS_TYPE} bus from {SRC} to {DEST}",
    "get me a {SEAT_TYPE} from {SRC} to {DEST} {DATE}",
    "i want to go from {SRC} to {DEST} {DATE} {TIME}",
    
    # Questions
    "is there a bus from {SRC} to {DEST}",
    "any {BUS_TYPE} bus from {SRC} to {DEST}",
    "which buses go from {SRC} to {DEST}",
    "what buses available from {SRC} to {DEST} {DATE}",
    "any {OPERATOR} {BUS_TYPE} bus from {SRC} to {DEST} {DATE} {TIME}",
]
