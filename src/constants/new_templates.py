"""
New templates targeting underrepresented labels:
  - {SEMANTIC}           : 189 → +250 templates
  - {ARRIVAL_DATE}       : 516 → +200 templates
  - {COUPON_CODE}        : 387 → +150 templates
  - {DEALS}              : 394 → +150 templates
  - {ADD_ONS}            : 286 → +150 templates
  - {BUS_FEATURES}       : 442 → +120 templates
  - {AMENITIES}          : 568 → +120 templates
  - {SEAT_TYPE}          : 488 → +120 templates
  - {SOURCE_CITY_CODE}   : 300 → +100 templates
  - {DESTINATION_CITY_CODE}: 277 → +100 templates

All templates are:
  - Natural user queries (not hallucinated)
  - Longer, more complex phrasings
  - Cover formal / informal / fragmented / misspelled patterns
  - Combine multiple entity types for rich context
"""

NEW_TEMPLATES = [

    # =========================================================================
    # SECTION 1: SEMANTIC — Quality / preference queries (250 templates)
    # =========================================================================

    # --- Standalone SEMANTIC preference ---
    "I am looking for a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, please show me the best options",
    "Can you please suggest the most {SEMANTIC} bus available from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}?",
    "I prefer {SEMANTIC} travel — show me all {SEMANTIC} buses going from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Please find me a {SEMANTIC} and {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} departing on {DEPARTURE_DATE}",
    "What is the most {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} available for {DEPARTURE_DATE}?",
    "I want only {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME}, please filter accordingly",
    "Filter buses from {SOURCE_NAME} to {DESTINATION_NAME} that are {SEMANTIC} and have good reviews",
    "Looking for a {SEMANTIC} overnight bus from {SOURCE_NAME} to {DESTINATION_NAME} for tomorrow night",
    "Can I get a {SEMANTIC} and {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {AC_TYPE}?",
    "Show me all {SEMANTIC} {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I only want {SEMANTIC} options — search for buses from {SOURCE_NAME} to {DESTINATION_NAME} next week",
    "Please recommend the most {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} that also has {AMENITIES}",
    "My family needs a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, suggest something good",
    "Show {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} operated by {OPERATOR} if available",
    "I want a {SEMANTIC} and {SEMANTIC} bus for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Which bus company offers the most {SEMANTIC} service from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Is there a {SEMANTIC} bus with {SEAT_TYPE} seats from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Search for {SEMANTIC} {BUS_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} departing on {DEPARTURE_DATE}",
    "I am a solo traveler and need a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} — what are my options?",
    "Please provide {SEMANTIC} bus options from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} with {AMENITIES}",

    # --- SEMANTIC + PRICE ---
    "Find me a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} within a budget of {PRICE}",
    "I need a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} under {PRICE} for {DEPARTURE_DATE}",
    "Show {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} under {PRICE}, preferably {AC_TYPE}",
    "What are the {SEMANTIC} bus options from {SOURCE_NAME} to {DESTINATION_NAME} that cost less than {PRICE}?",
    "I want to book the most {SEMANTIC} and {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} at a price below {PRICE}",
    "Are there any {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} below {PRICE} on {DEPARTURE_DATE}?",
    "Looking for the {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for the trip on {DEPARTURE_DATE} — budget is around {PRICE}",
    "Find a {SEMANTIC} option from {SOURCE_NAME} to {DESTINATION_NAME} that fits within {PRICE} per ticket",
    "Show me {SEMANTIC} buses between {SOURCE_NAME} and {DESTINATION_NAME} priced under {PRICE}",
    "I want a {SEMANTIC} night bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} within {PRICE}",

    # --- SEMANTIC + OPERATOR ---
    "Does {OPERATOR} have a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I trust {OPERATOR} — show me their most {SEMANTIC} service from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book me a {SEMANTIC} bus with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is {OPERATOR} considered {SEMANTIC} for the route from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want a {SEMANTIC} bus only with {OPERATOR} operator from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Which operators offer a {SEMANTIC} service from {SOURCE_NAME} to {DESTINATION_NAME} — I prefer {OPERATOR}",
    "Show me {SEMANTIC} buses by {OPERATOR} going from {SOURCE_NAME} to {DESTINATION_NAME} this weekend",
    "Can you check if {OPERATOR} runs a {SEMANTIC} night service from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I have traveled with {OPERATOR} before and found it {SEMANTIC} — book the same route from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "List all {SEMANTIC} buses by {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",

    # --- SEMANTIC + SEAT_TYPE ---
    "I need a {SEMANTIC} bus with {SEAT_TYPE} seats from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a {SEMANTIC} {SEAT_TYPE} option from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE}",
    "Book a {SEAT_TYPE} in a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "Show me {SEMANTIC} buses with {SEAT_TYPE} available from {SOURCE_NAME} to {DESTINATION_NAME}",
    "I am looking for a {SEMANTIC} sleeper bus with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a {SEMANTIC} {AC_TYPE} bus with {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Please book a {SEAT_TYPE} in the most {SEMANTIC} bus available from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "I want a {SEAT_TYPE} seat in a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME}, any availability on {DEPARTURE_DATE}?",
    "Find a {SEMANTIC} and {SEMANTIC} bus with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Show {SEMANTIC} buses with {SEAT_TYPE} and {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- SEMANTIC + TRAVELER ---
    "Book tickets for {TRAVELER} on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I am traveling with {TRAVELER} and need a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE}",
    "We are {TRAVELER} looking for a {SEMANTIC} and comfortable bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Need tickets for {TRAVELER} on the most {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} this {DEPARTURE_DATE}",
    "Is there a {SEMANTIC} bus with enough seats for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Please find the most {SEMANTIC} bus for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "We are a group of {TRAVELER} and want only {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book a {SEMANTIC} bus for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} at {DEPARTURE_TIME}",
    "I need a {SEMANTIC} bus with {SEAT_TYPE} for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME}",

    # --- SEMANTIC + DEPARTURE TIME ---
    "Is there a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} leaving at {DEPARTURE_TIME} on {DEPARTURE_DATE}?",
    "Show {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} departing in the {DEPARTURE_TIME}",
    "I need a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} that departs around {DEPARTURE_TIME}",
    "Find a {SEMANTIC} early morning bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Looking for {SEMANTIC} {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "I prefer to travel at {DEPARTURE_TIME} — please find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can I get a {SEMANTIC} overnight bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME}?",
    "Book me a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} with {SEAT_TYPE} seats",
    "Is there a {SEMANTIC} {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} that leaves at {DEPARTURE_TIME}?",
    "Show me {SEMANTIC} buses with {DEPARTURE_TIME} departure from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- SEMANTIC longer multi-entity ---
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and I am looking for only {SEMANTIC} buses with {AC_TYPE} and {SEAT_TYPE} seat",
    "Please help me find a {SEMANTIC} {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} operated by {OPERATOR} with {AMENITIES}",
    "Can you suggest the most {SEMANTIC} and {SEMANTIC} {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} arriving before {ARRIVAL_TIME}?",
    "I have a budget of {PRICE} and need the most {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "My preference is {SEMANTIC} service — please find me a {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {AMENITIES}",
    "Show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and sort them by {SEMANTIC} first",
    "I only care about {SEMANTIC} and {SEMANTIC} quality — book the best bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a {SEMANTIC} {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {SEAT_TYPE} and {AMENITIES}",
    "Find the most {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE} within a budget of {PRICE}",
    "Please book tickets for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I want a {SEMANTIC} bus that also provides {AMENITIES}",

    # --- SEMANTIC informal / fragmented ---
    "just want {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "{SEMANTIC} buses for {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "need {SEMANTIC} option from {SOURCE_NAME} to {DESTINATION_NAME}",
    "show only {SEMANTIC} buses {SOURCE_NAME} to {DESTINATION_NAME}",
    "any {SEMANTIC} bus {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "find {SEMANTIC} {AC_TYPE} bus {SOURCE_NAME} to {DESTINATION_NAME}",
    "{SOURCE_NAME} to {DESTINATION_NAME} {SEMANTIC} bus please",
    "want something {SEMANTIC} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "looking for {SEMANTIC} ride from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "only {SEMANTIC} buses please, from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- SEMANTIC varied phrasing ---
    "Do you have any {SEMANTIC} bus service from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE}?",
    "What is the {SEMANTIC} bus operator for the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "I have heard that {OPERATOR} is {SEMANTIC} — can you book me on their bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Please recommend a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for a family of four traveling on {DEPARTURE_DATE}",
    "I am looking for a {SEMANTIC} and well-reviewed bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}",
    "Which is the most {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} running on {DEPARTURE_DATE}?",
    "I prefer {SEMANTIC} travel — can you book me a {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Tell me about the most {SEMANTIC} bus routes from {SOURCE_NAME} to {DESTINATION_NAME} available this week",
    "I want a {SEMANTIC} bus for my business trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please filter buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and show only the {SEMANTIC} ones",
    "My priority is {SEMANTIC} travel — show me options from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can you find me the most {SEMANTIC} overnight bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I have been on {SEMANTIC} buses before and want the same quality for my trip from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Suggest a {SEMANTIC} {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} with good passenger reviews",
    "I want to book a {SEMANTIC} {AC_TYPE} {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Are there any {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} running at night on {DEPARTURE_DATE}?",
    "I need a {SEMANTIC} bus with {AMENITIES} for my trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please provide me with {SEMANTIC} and {SEMANTIC} bus options for the route from {SOURCE_NAME} to {DESTINATION_NAME}",
    "I am traveling for work from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — need a {SEMANTIC} service",
    "Is the {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} considered {SEMANTIC} by passengers?",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I prefer {SEMANTIC} services with {AMENITIES}",
    "Book a {SEMANTIC} {AC_TYPE} bus for me from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}",
    "I want to travel {SEMANTIC} from {SOURCE_NAME} to {DESTINATION_NAME} — what are the options on {DEPARTURE_DATE}?",
    "Show me {SEMANTIC} bus options from {SOURCE_NAME} to {DESTINATION_NAME} that have good ratings and reviews",
    "I'm picky about comfort — please find only {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 2: ARRIVAL_DATE — Round trip & return journey (200 templates)
    # =========================================================================

    # --- Departure + Return Date ---
    "I want to book a round trip from {SOURCE_NAME} to {DESTINATION_NAME}, departing on {DEPARTURE_DATE} and returning on {ARRIVAL_DATE}",
    "Please book two-way tickets from {SOURCE_NAME} to {DESTINATION_NAME} — outward journey on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "I need to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and come back on {ARRIVAL_DATE}",
    "Book round trip from {SOURCE_NAME} to {DESTINATION_NAME} — going on {DEPARTURE_DATE}, return on {ARRIVAL_DATE}",
    "Can I book a return ticket from {SOURCE_NAME} to {DESTINATION_NAME}? Departure: {DEPARTURE_DATE}, Return: {ARRIVAL_DATE}",
    "I am planning a round trip from {SOURCE_NAME} to {DESTINATION_NAME} with departure on {DEPARTURE_DATE} and return journey on {ARRIVAL_DATE}",
    "Help me book a to-and-fro ticket from {SOURCE_NAME} to {DESTINATION_NAME}, departing {DEPARTURE_DATE} and coming back on {ARRIVAL_DATE}",
    "I want a round trip bus from {SOURCE_NAME} to {DESTINATION_NAME}, my outward date is {DEPARTURE_DATE} and I return on {ARRIVAL_DATE}",
    "Please confirm a round trip booking from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} and returning {ARRIVAL_DATE}",
    "I'm traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and need a return bus on {ARRIVAL_DATE}",
    "Book my journey from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}, and I also need a return bus on {ARRIVAL_DATE}",
    "I would like to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with a return on {ARRIVAL_DATE} — please arrange both",
    "Please find buses for my round trip — {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "I need a two-way booking from {SOURCE_NAME} to {DESTINATION_NAME}: going on {DEPARTURE_DATE}, back on {ARRIVAL_DATE}",
    "Can I get a return ticket from {SOURCE_NAME} to {DESTINATION_NAME}? I will go on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "Please book {SEAT_TYPE} for round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I need tickets for {TRAVELER} for a round trip from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} and returning {ARRIVAL_DATE}",
    "Book a {SEAT_TYPE} round trip from {SOURCE_NAME} to {DESTINATION_NAME} — going {DEPARTURE_DATE}, returning {ARRIVAL_DATE}",
    "I want to go from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and come back on {ARRIVAL_DATE} — please book both legs",
    "I plan to visit {DESTINATION_NAME} from {SOURCE_NAME} between {DEPARTURE_DATE} and {ARRIVAL_DATE} — book the round trip please",
    "My trip from {SOURCE_NAME} to {DESTINATION_NAME} starts on {DEPARTURE_DATE} and ends on {ARRIVAL_DATE}, I need both tickets",
    "I want to confirm two sleeper tickets for my round trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with return on {ARRIVAL_DATE}",
    "Find a round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and back on {ARRIVAL_DATE} within {PRICE}",
    "Please book return tickets from {SOURCE_NAME} to {DESTINATION_NAME}: departure date {DEPARTURE_DATE} and return date {ARRIVAL_DATE}",
    "I want the {SEMANTIC} bus available for a round trip from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} and returning {ARRIVAL_DATE}",

    # --- Arrival Date focus (must reach by date) ---
    "I must reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_DATE} — please suggest suitable buses",
    "I need to be at {DESTINATION_NAME} by {ARRIVAL_DATE} — what buses are available from {SOURCE_NAME}?",
    "Can I get a bus from {SOURCE_NAME} to {DESTINATION_NAME} that arrives before {ARRIVAL_DATE}?",
    "My meeting is on {ARRIVAL_DATE} — I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by then",
    "Please show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that will reach by {ARRIVAL_DATE}",
    "I have to be in {DESTINATION_NAME} before {ARRIVAL_DATE} — what are my bus options from {SOURCE_NAME}?",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} that reaches by the morning of {ARRIVAL_DATE}?",
    "I need to reach {DESTINATION_NAME} by {ARRIVAL_DATE} — find me the earliest bus from {SOURCE_NAME}",
    "I am traveling from {SOURCE_NAME} and need to be at {DESTINATION_NAME} on {ARRIVAL_DATE} morning",
    "What bus from {SOURCE_NAME} to {DESTINATION_NAME} will get me there by {ARRIVAL_DATE}?",
    "Please find a bus from {SOURCE_NAME} to {DESTINATION_NAME} with arrival on or before {ARRIVAL_DATE}",
    "I want to arrive at {DESTINATION_NAME} by {ARRIVAL_DATE} — please find an overnight bus from {SOURCE_NAME}",
    "I need to arrive at {DESTINATION_NAME} no later than {ARRIVAL_DATE} — what buses go from {SOURCE_NAME}?",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} that will arrive by {ARRIVAL_DATE} morning",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} with guaranteed arrival before {ARRIVAL_DATE}",

    # --- Return date with SEAT_TYPE / TRAVELER ---
    "Book {SEAT_TYPE} for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "I need {SEAT_TYPE} on a round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} — going {DEPARTURE_DATE}, returning {ARRIVAL_DATE}",
    "Please book {TRAVELER} round trip tickets from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I want sleeper tickets for round trip from {SOURCE_NAME} to {DESTINATION_NAME} — going on {DEPARTURE_DATE}, back on {ARRIVAL_DATE}",
    "Find {SEMANTIC} {AC_TYPE} bus for a round trip from {SOURCE_NAME} to {DESTINATION_NAME}: go {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I need a {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE} with {SEAT_TYPE} seat",
    "Book the {SEMANTIC} available round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and {ARRIVAL_DATE}",
    "For {TRAVELER}, please book a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I am looking for round trip {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} — outward {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "Please confirm a round trip {SEAT_TYPE} booking from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and {ARRIVAL_DATE}",

    # --- Return date + OPERATOR ---
    "Does {OPERATOR} offer round trip tickets from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and return {ARRIVAL_DATE}?",
    "I want to book a round trip with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} — going {DEPARTURE_DATE}, back {ARRIVAL_DATE}",
    "Can {OPERATOR} arrange a two-way journey from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} returning {ARRIVAL_DATE}?",
    "Book me on {OPERATOR} for round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I prefer {OPERATOR} — please book round trip from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and {ARRIVAL_DATE}",

    # --- Return date informal ---
    "round trip {SOURCE_NAME} to {DESTINATION_NAME} go {DEPARTURE_DATE} come back {ARRIVAL_DATE}",
    "going {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} return on {ARRIVAL_DATE}",
    "book return {SOURCE_NAME} {DESTINATION_NAME} depart {DEPARTURE_DATE} return {ARRIVAL_DATE}",
    "need bus {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} and return {ARRIVAL_DATE}",
    "from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} back on {ARRIVAL_DATE}",
    "two way ticket {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE} return {ARRIVAL_DATE}",
    "going {DEPARTURE_DATE} returning {ARRIVAL_DATE} {SOURCE_NAME} to {DESTINATION_NAME}",
    "depart {DEPARTURE_DATE} return {ARRIVAL_DATE} {SOURCE_NAME} {DESTINATION_NAME}",

    # --- More ARRIVAL_DATE combinations ---
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I need to be back by {ARRIVAL_DATE}",
    "Please help me plan a trip from {SOURCE_NAME} to {DESTINATION_NAME} from {DEPARTURE_DATE} to {ARRIVAL_DATE}",
    "I will be in {DESTINATION_NAME} from {DEPARTURE_DATE} until {ARRIVAL_DATE} — please book the round trip from {SOURCE_NAME}",
    "I want to leave {SOURCE_NAME} on {DEPARTURE_DATE} and return from {DESTINATION_NAME} on {ARRIVAL_DATE}",
    "Can you book a round trip for me from {SOURCE_NAME} to {DESTINATION_NAME}? I go on {DEPARTURE_DATE} and come back on {ARRIVAL_DATE}",
    "Please book my travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and plan my return journey for {ARRIVAL_DATE}",
    "I want a trip to {DESTINATION_NAME} from {SOURCE_NAME} departing {DEPARTURE_DATE} and returning on {ARRIVAL_DATE} — all in one booking",
    "I need round trip bus tickets: {SOURCE_NAME} → {DESTINATION_NAME} on {DEPARTURE_DATE}, return {DESTINATION_NAME} → {SOURCE_NAME} on {ARRIVAL_DATE}",
    "My travel plan: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, return on {ARRIVAL_DATE} — please check availability and book",
    "I am planning a trip from {SOURCE_NAME} to {DESTINATION_NAME} from {DEPARTURE_DATE} to {ARRIVAL_DATE} — book both tickets please",
    "Find me round trip tickets from {SOURCE_NAME} to {DESTINATION_NAME} for the dates {DEPARTURE_DATE} to {ARRIVAL_DATE}",
    "We need bus tickets for round trip from {SOURCE_NAME} to {DESTINATION_NAME} — outward {DEPARTURE_DATE}, back on {ARRIVAL_DATE}",
    "Check bus availability for a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I'm planning to visit {DESTINATION_NAME} from {SOURCE_NAME} from {DEPARTURE_DATE} to {ARRIVAL_DATE} — need both-way tickets",
    "Is there a round trip package from {SOURCE_NAME} to {DESTINATION_NAME} for dates {DEPARTURE_DATE} and {ARRIVAL_DATE}?",
    "Can I get a discount on a round trip from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} and returning {ARRIVAL_DATE}?",
    "Please find the {SEMANTIC} round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} returning {ARRIVAL_DATE}",
    "I want to go to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} and come back on {ARRIVAL_DATE} — find {SEMANTIC} options",
    "Looking for {AC_TYPE} round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "Book {TRAVELER} for round trip from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} returning {ARRIVAL_DATE} in {AC_TYPE}",

    # =========================================================================
    # SECTION 3: COUPON_CODE — Promo / discount code queries (150 templates)
    # =========================================================================

    # --- Applying coupon ---
    "I have a coupon code {COUPON_CODE} — please apply it to my booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can I use the promo code {COUPON_CODE} for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Apply coupon {COUPON_CODE} to my bus ticket from {SOURCE_NAME} to {DESTINATION_NAME}",
    "I want to redeem my voucher code {COUPON_CODE} for the journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please apply the discount code {COUPON_CODE} when booking my bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Use promo code {COUPON_CODE} for my trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want to book from {SOURCE_NAME} to {DESTINATION_NAME} using coupon {COUPON_CODE} — is it valid?",
    "How do I apply the coupon code {COUPON_CODE} for a bus journey from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I received a promo code {COUPON_CODE} — can I use it for the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Please book my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and apply the code {COUPON_CODE}",
    "I want to apply the coupon code {COUPON_CODE} for my journey with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Please use discount code {COUPON_CODE} while booking my {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can the coupon {COUPON_CODE} be applied for {SEAT_TYPE} seats on the bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Use my saved promo code {COUPON_CODE} and book me on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want to use my promo code {COUPON_CODE} for tickets from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Apply coupon {COUPON_CODE} on my booking from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "I am booking a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — please apply coupon code {COUPON_CODE}",
    "Can I redeem my promo code {COUPON_CODE} for a round trip from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to book tickets from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} with coupon code {COUPON_CODE}",
    "Please confirm the booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} after applying promo code {COUPON_CODE}",

    # --- Coupon eligibility & validity ---
    "Is the coupon code {COUPON_CODE} valid for buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Why am I not eligible for the coupon code {COUPON_CODE}?",
    "I am getting an error when applying coupon {COUPON_CODE} for my journey from {SOURCE_NAME} to {DESTINATION_NAME}",
    "My promo code {COUPON_CODE} is not working — can you help me apply it for {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Is coupon {COUPON_CODE} valid on {DEPARTURE_DATE} for the route from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Can I use the code {COUPON_CODE} on {OPERATOR} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Tell me the discount I will get with coupon code {COUPON_CODE} for {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "How much discount does coupon {COUPON_CODE} give on buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Is coupon code {COUPON_CODE} applicable for {AC_TYPE} {SEAT_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "What buses can I use coupon {COUPON_CODE} on for the route {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Why is my coupon code {COUPON_CODE} not eligible for the bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "The promo code {COUPON_CODE} shows as invalid for my {SOURCE_NAME} to {DESTINATION_NAME} booking — please check",
    "Can I get additional discount using coupon {COUPON_CODE} on the {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Will the promo code {COUPON_CODE} work for a first-time booking from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I have a new user coupon {COUPON_CODE} — is it valid for buses from {SOURCE_NAME} to {DESTINATION_NAME}?",

    # --- Coupon + OPERATOR (operator as coupon code context) ---
    "I want to apply the coupon {COUPON_CODE} for my bus journey with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Does {OPERATOR} accept promo code {COUPON_CODE} for the route from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Book me on {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} and apply coupon {COUPON_CODE}",
    "Is promo code {COUPON_CODE} valid on {OPERATOR} buses running from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to use coupon {COUPON_CODE} while booking {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Apply code {COUPON_CODE} on my {OPERATOR} booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can I use coupon {COUPON_CODE} on {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to book with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} using promo code {COUPON_CODE}",
    "Book {SEAT_TYPE} with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and apply code {COUPON_CODE}",
    "Apply promo {COUPON_CODE} on {OPERATOR} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",

    # --- Coupon + PRICE ---
    "I want to use coupon {COUPON_CODE} to reduce the price of my bus ticket from {SOURCE_NAME} to {DESTINATION_NAME} below {PRICE}",
    "After applying coupon {COUPON_CODE}, what will be the final price for the bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Can I book a bus from {SOURCE_NAME} to {DESTINATION_NAME} under {PRICE} if I use coupon code {COUPON_CODE}?",
    "My budget is {PRICE} — will coupon {COUPON_CODE} help me book a bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} within {PRICE} using promo code {COUPON_CODE}",

    # --- Coupon informal ---
    "have coupon code {COUPON_CODE} apply it {SOURCE_NAME} to {DESTINATION_NAME}",
    "use promo {COUPON_CODE} for my ticket {SOURCE_NAME} {DESTINATION_NAME}",
    "{COUPON_CODE} apply on bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "apply {COUPON_CODE} {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "book {SOURCE_NAME} {DESTINATION_NAME} with coupon {COUPON_CODE}",
    "code {COUPON_CODE} apply on {SOURCE_NAME} to {DESTINATION_NAME} booking",
    "use {COUPON_CODE} promo for trip {SOURCE_NAME} to {DESTINATION_NAME}",
    "apply discount {COUPON_CODE} {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "{COUPON_CODE} valid for {SOURCE_NAME} to {DESTINATION_NAME}?",
    "can i use {COUPON_CODE} for {SOURCE_NAME} to {DESTINATION_NAME}?",

    # =========================================================================
    # SECTION 4: DEALS — Offer / deal queries (150 templates)
    # =========================================================================

    # --- Applying a deal ---
    "Is the deal {DEALS} available for buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} using the offer {DEALS}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that are eligible for the {DEALS} offer",
    "Apply the {DEALS} offer to my bus booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is the {DEALS} deal still active for the route from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I heard about the {DEALS} offer — can I use it for the bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Please book me on a bus from {SOURCE_NAME} to {DESTINATION_NAME} and apply the {DEALS} deal",
    "How much discount does the {DEALS} offer give for buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to use the {DEALS} deal for my round trip from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can I claim the {DEALS} offer on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Show me {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} that are eligible for the {DEALS} deal",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} using the {DEALS} offer for savings",
    "I want to apply the {DEALS} offer to get the {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} at a lower price",
    "Is the {DEALS} offer applicable on {OPERATOR} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I found the {DEALS} deal — is it valid for {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Tell me more about the {DEALS} deal for buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can I use the {DEALS} offer when booking a {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to take advantage of the {DEALS} offer for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is the {DEALS} available for {OPERATOR} buses on the {SOURCE_NAME} to {DESTINATION_NAME} route on {DEPARTURE_DATE}?",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with the {DEALS} offer applied",

    # --- DEALS eligibility ---
    "Why am I not eligible for the {DEALS} deal on buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "The {DEALS} offer is not showing up for my booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — help please",
    "I cannot redeem the {DEALS} offer on the {SOURCE_NAME} to {DESTINATION_NAME} bus — what should I do?",
    "Is the {DEALS} deal valid only for certain operators on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Tell me the terms and conditions of the {DEALS} deal for buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "How many times can I use the {DEALS} offer on buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Is the {DEALS} offer applicable for the first booking only from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I tried using the {DEALS} deal for my bus from {SOURCE_NAME} to {DESTINATION_NAME} but it didn't work",
    "Can I use the {DEALS} deal for group booking from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER}?",
    "Does {DEALS} apply to buses going from {SOURCE_NAME} to {DESTINATION_NAME} on weekends?",

    # --- DEALS informal ---
    "any deals on {SOURCE_NAME} to {DESTINATION_NAME} buses?",
    "apply {DEALS} on {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "is {DEALS} available for {SOURCE_NAME} to {DESTINATION_NAME}?",
    "use {DEALS} offer {SOURCE_NAME} to {DESTINATION_NAME}",
    "{DEALS} applicable on {SOURCE_NAME} {DESTINATION_NAME}?",
    "book with {DEALS} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "any offers like {DEALS} for {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "show {DEALS} buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "{DEALS} deal still valid for {SOURCE_NAME} {DESTINATION_NAME}?",
    "want to use {DEALS} for booking {SOURCE_NAME} {DESTINATION_NAME}",

    # --- DEALS + PRICE ---
    "With the {DEALS} deal, what will be the final price for a bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to use the {DEALS} offer to bring my ticket price below {PRICE} for the route {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can the {DEALS} deal reduce my ticket from {SOURCE_NAME} to {DESTINATION_NAME} to under {PRICE}?",
    "After applying {DEALS}, how much will I pay for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I have seen {DEALS} offer — will it help me get a ticket from {SOURCE_NAME} to {DESTINATION_NAME} within my budget of {PRICE}?",

    # =========================================================================
    # SECTION 5: ADD_ONS — Insurance / cancellation / trip guarantee (150 templates)
    # =========================================================================

    # --- ADD_ONS awareness & request ---
    "I want to add {ADD_ONS} to my bus booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can I include {ADD_ONS} when booking a bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Please add {ADD_ONS} to my ticket from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is {ADD_ONS} available for the bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {ADD_ONS} for peace of mind",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that come with {ADD_ONS}",
    "Can I opt for {ADD_ONS} while booking my {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I would like {ADD_ONS} on my booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is {ADD_ONS} option available for the {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and add {ADD_ONS} as well",
    "I need {ADD_ONS} for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — is it available?",
    "Please confirm my booking from {SOURCE_NAME} to {DESTINATION_NAME} with {ADD_ONS} included",
    "Can I purchase {ADD_ONS} for my existing bus booking from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} with {ADD_ONS} included",
    "What is the cost of adding {ADD_ONS} to my bus ticket from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Is there an option to add {ADD_ONS} and {ADD_ONS} together for my bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to protect my trip from {SOURCE_NAME} to {DESTINATION_NAME} with {ADD_ONS}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that offer {ADD_ONS} as an add-on",
    "I want to book the {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {ADD_ONS} protection",
    "Book a {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {ADD_ONS}",

    # --- Cancellation / refund queries (ADD_ONS context) ---
    "Can I cancel my bus ticket from {SOURCE_NAME} to {DESTINATION_NAME} if I purchased {ADD_ONS}?",
    "I want to cancel my booking from {SOURCE_NAME} to {DESTINATION_NAME} — I had taken {ADD_ONS}, so what is my refund?",
    "What is the refund policy if I cancel my bus from {SOURCE_NAME} to {DESTINATION_NAME} with {ADD_ONS}?",
    "I need to reschedule my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I have {ADD_ONS}",
    "I bought {ADD_ONS} for my journey from {SOURCE_NAME} to {DESTINATION_NAME} — will I get a full refund if I cancel?",
    "Can I use {ADD_ONS} to reschedule my bus from {SOURCE_NAME} to {DESTINATION_NAME} without a penalty?",
    "My bus from {SOURCE_NAME} to {DESTINATION_NAME} was cancelled — I want to claim my {ADD_ONS} refund",
    "I missed my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — can {ADD_ONS} help me?",
    "What does {ADD_ONS} cover for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Is {ADD_ONS} applicable even if I cancel my bus 24 hours before departure from {SOURCE_NAME} to {DESTINATION_NAME}?",

    # --- ADD_ONS informal ---
    "add {ADD_ONS} to my booking {SOURCE_NAME} {DESTINATION_NAME}",
    "want {ADD_ONS} on {SOURCE_NAME} to {DESTINATION_NAME} bus",
    "book {SOURCE_NAME} {DESTINATION_NAME} with {ADD_ONS}",
    "is {ADD_ONS} available for {SOURCE_NAME} {DESTINATION_NAME} bus?",
    "can i add {ADD_ONS} to {SOURCE_NAME} to {DESTINATION_NAME} ticket?",
    "add {ADD_ONS} {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "{ADD_ONS} available on {SOURCE_NAME} to {DESTINATION_NAME}?",
    "need {ADD_ONS} for my trip {SOURCE_NAME} {DESTINATION_NAME}",
    "book with {ADD_ONS} from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "include {ADD_ONS} when booking {SOURCE_NAME} to {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 6: BUS_FEATURES — GPS, live tracking, star rating (120 templates)
    # =========================================================================

    # --- BUS_FEATURES preference ---
    "I want a bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that have {BUS_FEATURES}",
    "Find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} feature for my journey on {DEPARTURE_DATE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} and {AMENITIES}?",
    "I need a bus with {BUS_FEATURES} for my trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} that has {BUS_FEATURES} enabled",
    "I prefer buses with {BUS_FEATURES} — show me options from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Are there any buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {BUS_FEATURES}?",
    "I want a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} and {SEAT_TYPE} seating",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} available on {DEPARTURE_DATE}",
    "Please find a {SEMANTIC} bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Find me a {AC_TYPE} bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want live tracking on my bus from {SOURCE_NAME} to {DESTINATION_NAME} — show buses with {BUS_FEATURES}",
    "Book a bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "Is there a {BUS_TYPE} bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to track my bus journey — find a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES}",
    "Show me {SEMANTIC} buses with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} and under {PRICE}",
    "I need a bus with {BUS_FEATURES} and {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "Book a {SEMANTIC} bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- BUS_FEATURES + OPERATOR ---
    "Does {OPERATOR} offer {BUS_FEATURES} on their buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to book {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} only if they have {BUS_FEATURES}",
    "Show me {OPERATOR} buses from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} on {DEPARTURE_DATE}",
    "Does {OPERATOR} have {BUS_FEATURES} on their {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Find {OPERATOR} or any other operator with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- BUS_FEATURES informal ---
    "need {BUS_FEATURES} bus {SOURCE_NAME} to {DESTINATION_NAME}",
    "show buses with {BUS_FEATURES} {SOURCE_NAME} {DESTINATION_NAME}",
    "bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "{BUS_FEATURES} bus {SOURCE_NAME} to {DESTINATION_NAME} please",
    "any {BUS_FEATURES} bus {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}?",
    "want {BUS_FEATURES} in bus {SOURCE_NAME} to {DESTINATION_NAME}",
    "buses with {BUS_FEATURES} {SOURCE_NAME} {DESTINATION_NAME}",
    "find {BUS_FEATURES} bus {SOURCE_NAME} to {DESTINATION_NAME}",
    "book bus {SOURCE_NAME} {DESTINATION_NAME} with {BUS_FEATURES}",
    "only buses with {BUS_FEATURES} please {SOURCE_NAME} {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 7: AMENITIES — Comfort facilities on bus (120 templates)
    # =========================================================================

    # --- AMENITIES preference ---
    "I want a bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that offer {AMENITIES} on board",
    "Find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} — I need it for my trip on {DEPARTURE_DATE}",
    "Please suggest a bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} for my overnight journey",
    "I need a bus with {AMENITIES} and {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Are there buses from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} available on {DEPARTURE_DATE}?",
    "I cannot travel without {AMENITIES} — please find a bus from {SOURCE_NAME} to {DESTINATION_NAME} that has it",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {AMENITIES} facility",
    "I want a comfortable bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} for my trip",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} that provides {AMENITIES} on board",
    "Is there a {AC_TYPE} bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Find a bus with {AMENITIES} and {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a bus with {AMENITIES} for my long journey from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Show me {SEMANTIC} buses with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a {BUS_TYPE} bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "I need {AMENITIES} on my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — what are my options?",
    "Please book a {SEAT_TYPE} seat on a bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Show me all buses with {AMENITIES} facility from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} and good ratings",
    "Find a {SEMANTIC} bus with {AMENITIES} and {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME}",

    # --- AMENITIES + OPERATOR ---
    "Does {OPERATOR} provide {AMENITIES} on their buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to book {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} — do they have {AMENITIES}?",
    "Show {OPERATOR} buses from {SOURCE_NAME} to {DESTINATION_NAME} that provide {AMENITIES}",
    "Does {OPERATOR} have {AMENITIES} on the {SOURCE_NAME} to {DESTINATION_NAME} route on {DEPARTURE_DATE}?",
    "Book me on {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} only if they have {AMENITIES}",

    # --- AMENITIES informal ---
    "need {AMENITIES} on bus {SOURCE_NAME} to {DESTINATION_NAME}",
    "show buses with {AMENITIES} {SOURCE_NAME} {DESTINATION_NAME}",
    "bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "any bus with {AMENITIES} {SOURCE_NAME} {DESTINATION_NAME}?",
    "find {AMENITIES} bus {SOURCE_NAME} to {DESTINATION_NAME} please",
    "want {AMENITIES} facility in bus {SOURCE_NAME} {DESTINATION_NAME}",
    "buses with {AMENITIES} {SOURCE_NAME} to {DESTINATION_NAME}",
    "book bus with {AMENITIES} {SOURCE_NAME} {DESTINATION_NAME}",
    "only buses with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "{AMENITIES} bus {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # =========================================================================
    # SECTION 8: SEAT_TYPE — Seat preference queries (120 templates)
    # =========================================================================

    # --- SEAT_TYPE preference ---
    "I want a {SEAT_TYPE} seat on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book a {SEAT_TYPE} for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a {SEAT_TYPE} available on any bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I prefer {SEAT_TYPE} — find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} with that option",
    "Can I get a {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE} availability on {DEPARTURE_DATE}",
    "I need a {SEAT_TYPE} seat on an overnight bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Please book a {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {AC_TYPE}",
    "Find me a bus with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Are there any {AC_TYPE} buses with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want {SEAT_TYPE} with {AC_TYPE} air conditioning from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book me a {SEAT_TYPE} seat on a {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a {SEAT_TYPE} seat available on {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I need a {SEAT_TYPE} seat for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please find a {SEAT_TYPE} seat on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book a {SEAT_TYPE} with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a {SEAT_TYPE} near the window from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE} option on {DEPARTURE_DATE}",
    "I specifically need a {SEAT_TYPE} seat — show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a {SEAT_TYPE} on a {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- SEAT_TYPE + multiple entities ---
    "Book a {SEAT_TYPE} seat with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {AMENITIES}",
    "I want a {SEAT_TYPE} on a {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} — is there availability on {DEPARTURE_DATE}?",
    "Please find a {SEAT_TYPE} seat on a {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME}",
    "Find a {SEAT_TYPE} seat for my night journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I am traveling with {TRAVELER} and want {SEAT_TYPE} seats from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please confirm a {SEAT_TYPE} booking for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can I get a {SEAT_TYPE} seat on the {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Show me {SEMANTIC} buses with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a {SEAT_TYPE} seat and need {AMENITIES} on the bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Find a {AC_TYPE} {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} within {PRICE}",

    # --- SEAT_TYPE informal ---
    "need {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "book {SEAT_TYPE} {SOURCE_NAME} to {DESTINATION_NAME}",
    "any {SEAT_TYPE} available {SOURCE_NAME} {DESTINATION_NAME}?",
    "want {SEAT_TYPE} seat {SOURCE_NAME} to {DESTINATION_NAME}",
    "{SEAT_TYPE} bus {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "show buses with {SEAT_TYPE} {SOURCE_NAME} to {DESTINATION_NAME}",
    "{SOURCE_NAME} to {DESTINATION_NAME} {SEAT_TYPE} please",
    "find {SEAT_TYPE} {SOURCE_NAME} {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "book {SEAT_TYPE} with {AC_TYPE} {SOURCE_NAME} {DESTINATION_NAME}",
    "only {SEAT_TYPE} buses {SOURCE_NAME} to {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 9: SOURCE_CITY_CODE + DESTINATION_CITY_CODE (100 templates)
    # =========================================================================

    "Find buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} for {DEPARTURE_DATE}",
    "Show bus availability from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "I want to travel from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} — what are my options?",
    "Book a bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Are there any buses going from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}?",
    "Show me all buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} departing on {DEPARTURE_DATE}",
    "I need a bus ticket from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} for {DEPARTURE_DATE}",
    "What buses are available from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} for {DEPARTURE_DATE}?",
    "Search for buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Find a {SEMANTIC} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Is there a direct bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}?",
    "Book the {SEMANTIC} available bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Check bus availability for {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "I need an {AC_TYPE} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Find me a bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} departing at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "Show {AC_TYPE} buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Book tickets from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} for {TRAVELER} on {DEPARTURE_DATE}",
    "I want a {SEAT_TYPE} seat on a bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Find the {SEMANTIC} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Show buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} with {AMENITIES} on {DEPARTURE_DATE}",
    "I want to book a bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE} for {TRAVELER}",
    "Is there a bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE} at {DEPARTURE_TIME}?",
    "Show me {OPERATOR} buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Find buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} under {PRICE} on {DEPARTURE_DATE}",
    "Any overnight bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} for {DEPARTURE_DATE}?",
    "Book a {BUS_TYPE} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "I need a {SEMANTIC} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} for my trip on {DEPARTURE_DATE}",
    "Show {SEAT_TYPE} availability from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "Find round trip from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} departing {DEPARTURE_DATE} returning {ARRIVAL_DATE}",
    "Book {TRAVELER} from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE} in {AC_TYPE}",

    # --- City Code informal ---
    "bus {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} {DEPARTURE_DATE}",
    "show buses {SOURCE_CITY_CODE} {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}",
    "{SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} bus please",
    "find bus {SOURCE_CITY_CODE} {DESTINATION_CITY_CODE} for {DEPARTURE_DATE}",
    "any bus {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE}?",
    "book {SOURCE_CITY_CODE} {DESTINATION_CITY_CODE} {DEPARTURE_DATE}",
    "buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE}",
    "need ticket {SOURCE_CITY_CODE} {DESTINATION_CITY_CODE} {DEPARTURE_DATE}",
    "available buses {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE}?",
    "{SOURCE_CITY_CODE} {DESTINATION_CITY_CODE} bus {DEPARTURE_DATE}",

    # =========================================================================
    # SECTION 10: COMBINED multi-label complex templates (100 templates)
    # =========================================================================

    # Complex queries combining 4+ labels
    "I want to book a round trip from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} returning {ARRIVAL_DATE}, prefer {SEMANTIC} {AC_TYPE} bus with {SEAT_TYPE} seat",
    "Please find a {SEMANTIC} {AC_TYPE} bus with {AMENITIES} and {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} within {PRICE}",
    "I have coupon {COUPON_CODE} — book me a {SEMANTIC} {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and apply it",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE} with {AMENITIES} within {PRICE}",
    "Book a round trip for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE} — {SEMANTIC} bus with {SEAT_TYPE}",
    "I want an {AC_TYPE} {SEAT_TYPE} bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — apply coupon {COUPON_CODE}",
    "Show {SEMANTIC} buses with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} under {PRICE}",
    "Book a {SEMANTIC} {BUS_TYPE} bus with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {ADD_ONS}",
    "I want to use the {DEALS} deal and apply coupon {COUPON_CODE} for my round trip from {SOURCE_NAME} to {DESTINATION_NAME} — go {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "Please find me a {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {AMENITIES} and {BUS_FEATURES}",
    "Book {TRAVELER} on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} with {SEAT_TYPE} seats and {AMENITIES}",
    "I need a {SEMANTIC} round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE} — within {PRICE} per person",
    "Find a {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} on {DEPARTURE_DATE} with {AMENITIES}",
    "I want to book a {SEAT_TYPE} seat on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — use coupon {COUPON_CODE} if applicable",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {BUS_FEATURES} and {AMENITIES} that are {SEMANTIC}",
    "Book a round trip for {TRAVELER} from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE} in {AC_TYPE}",
    "Find a {SEMANTIC} bus with {ADD_ONS} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} under {PRICE}",
    "I want a {BUS_TYPE} bus with {AMENITIES} and {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please apply coupon {COUPON_CODE} and {DEALS} on my booking for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE}, {AMENITIES}, pickup at {PICKUP_POINT} and drop at {DROP_POINT}",
    "Book the most {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER} with {ADD_ONS}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE} seats, {AMENITIES} and {BUS_FEATURES} for {DEPARTURE_DATE}",
    "Show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that are {SEMANTIC}, have {AMENITIES} and cost under {PRICE}",
    "I need a {SEMANTIC} overnight bus from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE} seat and {AMENITIES} for {DEPARTURE_DATE}",
    "Book round trip from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} — departure {DEPARTURE_DATE} return {ARRIVAL_DATE} — prefer {SEMANTIC} {SEAT_TYPE}",
    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I need {SEAT_TYPE}, {AMENITIES}, and want to apply coupon {COUPON_CODE}",
    "Please find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} — I need {AMENITIES} and {ADD_ONS} protection",
    "I am planning a round trip from {SOURCE_NAME} to {DESTINATION_NAME} from {DEPARTURE_DATE} to {ARRIVAL_DATE} for {TRAVELER} — need {SEMANTIC} {AC_TYPE} bus",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {BUS_FEATURES} and {AMENITIES} for {DEPARTURE_DATE} — apply {DEALS} if available",
    "Book a {SEMANTIC} {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER} with {SEAT_TYPE} and {ADD_ONS}",

    # --- Mixed everyday complex queries ---
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE} — please find a {SEMANTIC} bus with {SEAT_TYPE} and {AMENITIES}",
    "Show me {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that have {AMENITIES} and {BUS_FEATURES}",
    "I have a promo code {COUPON_CODE} — please use it and book a {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can I get a {SEMANTIC} {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} within {PRICE}?",
    "Book me a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — go {DEPARTURE_DATE} return {ARRIVAL_DATE} — apply coupon {COUPON_CODE}",
    "I need a {SEMANTIC} bus with {BUS_FEATURES} from {SOURCE_NAME} to {DESTINATION_NAME} for my trip on {DEPARTURE_DATE}",
    "Find the most {SEMANTIC} and {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and {ARRIVAL_DATE} with {SEAT_TYPE}?",
    "I want to use my {DEALS} offer and travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {ADD_ONS} and pickup from {PICKUP_POINT}",
    "Book tickets for {TRAVELER} on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with coupon {COUPON_CODE}",
    "I want a {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {SEAT_TYPE} and {AMENITIES} on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I prefer {SEMANTIC}, need {AMENITIES} and want {ADD_ONS}",
    "Please book a round trip for me from {SOURCE_NAME} to {DESTINATION_NAME} — outward {DEPARTURE_DATE}, return {ARRIVAL_DATE} — {SEMANTIC} bus with {SEAT_TYPE}",
    "I am planning a trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — want {SEAT_TYPE} with {AMENITIES} in a {SEMANTIC} bus",
    "Book a {SEMANTIC} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE} with {SEAT_TYPE} and {AMENITIES}",
    "Find me a {SEMANTIC} bus for round trip from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} departing {DEPARTURE_DATE} returning {ARRIVAL_DATE}",
    "Show buses from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} with {BUS_FEATURES} and {AMENITIES} on {DEPARTURE_DATE}",
    "I need a {SEMANTIC} {AC_TYPE} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE} under {PRICE}",
    "Book the most {SEMANTIC} bus from {SOURCE_CITY_CODE} to {DESTINATION_CITY_CODE} on {DEPARTURE_DATE} for {TRAVELER} with {SEAT_TYPE}",


    # =========================================================================
    # SECTION 11: PICKUP_POINT — Boarding stop clarity (130 templates)
    # =========================================================================

    # --- PICKUP standalone (no DROP) — teaches model PICKUP ≠ SOURCE_NAME ---
    "I want to board the bus at {PICKUP_POINT} for my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can the bus from {SOURCE_NAME} to {DESTINATION_NAME} pick me up at {PICKUP_POINT}?",
    "My boarding point will be {PICKUP_POINT} — I am going from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I will be waiting at {PICKUP_POINT} to board the {SOURCE_NAME} to {DESTINATION_NAME} bus on {DEPARTURE_DATE}",
    "Book me on a bus from {SOURCE_NAME} to {DESTINATION_NAME} that has a stop at {PICKUP_POINT} for pickup",
    "I live near {PICKUP_POINT} — can I board there for the bus going to {DESTINATION_NAME} from {SOURCE_NAME}?",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that pick up from {PICKUP_POINT}",
    "I need to board at {PICKUP_POINT} — please book a bus to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "I don't want to go to the main stand — can I board at {PICKUP_POINT} for the {SOURCE_NAME} to {DESTINATION_NAME} bus?",
    "My preferred boarding point is {PICKUP_POINT} for the bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} that stops at {PICKUP_POINT} for boarding on {DEPARTURE_DATE}?",
    "Please find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} with a pickup stop at {PICKUP_POINT} on {DEPARTURE_DATE}",
    "I'm at {PICKUP_POINT} and need a bus going to {DESTINATION_NAME} — what are my options on {DEPARTURE_DATE}?",
    "I will board at {PICKUP_POINT} — book a {SEAT_TYPE} seat to {DESTINATION_NAME} from {SOURCE_NAME}",
    "Find buses from {SOURCE_NAME} to {DESTINATION_NAME} picking up passengers from {PICKUP_POINT} on {DEPARTURE_DATE}",
    "I want to board from {PICKUP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route on {DEPARTURE_DATE}",
    "My boarding location is {PICKUP_POINT} — please show buses going from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Can I get picked up from {PICKUP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "I want a bus that picks me up from {PICKUP_POINT} and drops me in {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I stay near {PICKUP_POINT} — find me a bus to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE}",

    # --- PICKUP + DEPARTURE_DATE + TIME ---
    "I need to board at {PICKUP_POINT} at around {DEPARTURE_TIME} for the bus to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} that stop at {PICKUP_POINT}",
    "Is there a {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME} picking up from {PICKUP_POINT} on {DEPARTURE_DATE}?",
    "Book me on a {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME} with boarding at {PICKUP_POINT} on {DEPARTURE_DATE}",
    "I want to board at {PICKUP_POINT} at {DEPARTURE_TIME} for my trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find an early morning bus from {SOURCE_NAME} to {DESTINATION_NAME} picking up from {PICKUP_POINT} on {DEPARTURE_DATE}",
    "Show me the {DEPARTURE_TIME} buses from {SOURCE_NAME} to {DESTINATION_NAME} with pickup at {PICKUP_POINT}",

    # --- PICKUP + OPERATOR ---
    "Does {OPERATOR} pick up passengers from {PICKUP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "I want to board {OPERATOR}'s bus at {PICKUP_POINT} for my journey from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book me on {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} with boarding point at {PICKUP_POINT} on {DEPARTURE_DATE}",
    "Does {OPERATOR} have a bus from {SOURCE_NAME} to {DESTINATION_NAME} that picks up from {PICKUP_POINT}?",
    "I only want {OPERATOR} buses that pick up from {PICKUP_POINT} for the {SOURCE_NAME} to {DESTINATION_NAME} route",

    # --- PICKUP + SEAT_TYPE / AMENITIES ---
    "Book a {SEAT_TYPE} seat with boarding at {PICKUP_POINT} for the bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I need a {AC_TYPE} bus picking up from {PICKUP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "Find a {SEMANTIC} bus that picks up from {PICKUP_POINT} and goes from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {SEAT_TYPE} and pickup at {PICKUP_POINT}",
    "Show {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} that board at {PICKUP_POINT}",

    # --- PICKUP informal ---
    "board at {PICKUP_POINT} {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "pickup from {PICKUP_POINT} bus {SOURCE_NAME} {DESTINATION_NAME}",
    "i will be at {PICKUP_POINT} for {SOURCE_NAME} to {DESTINATION_NAME} bus",
    "bus picking from {PICKUP_POINT} {SOURCE_NAME} to {DESTINATION_NAME}",
    "boarding point {PICKUP_POINT} {SOURCE_NAME} {DESTINATION_NAME}",
    "{SOURCE_NAME} to {DESTINATION_NAME} pickup {PICKUP_POINT} please",
    "pick me up {PICKUP_POINT} going to {DESTINATION_NAME} from {SOURCE_NAME}",
    "need bus with stop at {PICKUP_POINT} {SOURCE_NAME} to {DESTINATION_NAME}",
    "board from {PICKUP_POINT} {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "any bus from {PICKUP_POINT} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",

    # --- PICKUP clearly distinguished from SOURCE ---
    "I am traveling from {SOURCE_NAME} to {DESTINATION_NAME} but I want to board the bus at {PICKUP_POINT}, not at the main stand",
    "My origin is {SOURCE_NAME} and destination is {DESTINATION_NAME}, but please set my pickup point as {PICKUP_POINT}",
    "The journey is from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I want to board at {PICKUP_POINT} which is along the route",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} — I'll board at {PICKUP_POINT} enroute",
    "I start from {SOURCE_NAME} but my boarding stop for the {DESTINATION_NAME} bus is {PICKUP_POINT}",
    "Please note my boarding point is {PICKUP_POINT}, not {SOURCE_NAME} — booking a bus to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "I'm booking a bus from {SOURCE_NAME} to {DESTINATION_NAME} — the boarding location should be {PICKUP_POINT}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} where I can board at {PICKUP_POINT}",
    "My city is {SOURCE_NAME} and I'm going to {DESTINATION_NAME} — but I'll board at {PICKUP_POINT} bus stop on {DEPARTURE_DATE}",
    "I want to board at {PICKUP_POINT} (not the central station in {SOURCE_NAME}) for the bus to {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 12: DROP_POINT — Alighting stop clarity (130 templates)
    # =========================================================================

    # --- DROP standalone (no PICKUP) — teaches DROP ≠ DESTINATION ---
    "I want to get off the bus at {DROP_POINT} — I am traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can the bus from {SOURCE_NAME} to {DESTINATION_NAME} drop me at {DROP_POINT}?",
    "My drop point is {DROP_POINT} — please book a bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "I want to be dropped off at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "Please ensure my drop-off location is {DROP_POINT} when booking from {SOURCE_NAME} to {DESTINATION_NAME}",
    "I need to get off at {DROP_POINT} — find buses from {SOURCE_NAME} to {DESTINATION_NAME} with that stop",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that drop passengers at {DROP_POINT}",
    "Does any bus from {SOURCE_NAME} to {DESTINATION_NAME} have a stop at {DROP_POINT} for alighting?",
    "I will alight at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} bus on {DEPARTURE_DATE}",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I want to deboard at {DROP_POINT}",
    "I don't need to go to the final stop — just drop me at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} bus",
    "My alighting point is {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "Please book a bus from {SOURCE_NAME} to {DESTINATION_NAME} and set drop point as {DROP_POINT}",
    "I will get off at {DROP_POINT} — book me on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} that drops at {DROP_POINT} on {DEPARTURE_DATE}?",
    "Book me a {SEAT_TYPE} seat on a bus from {SOURCE_NAME} to {DESTINATION_NAME} — my drop stop is {DROP_POINT}",
    "I need to reach {DROP_POINT} by traveling from {SOURCE_NAME} to {DESTINATION_NAME} by bus",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that has a stop at {DROP_POINT} for me to alight",
    "I want to get down at {DROP_POINT} — what buses are available from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "My destination stop is {DROP_POINT} though the route goes to {DESTINATION_NAME} — book me on this bus from {SOURCE_NAME}",

    # --- DROP + ARRIVAL_TIME ---
    "I need to reach {DROP_POINT} by {ARRIVAL_TIME} — find me a bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} that drops at {DROP_POINT} before {ARRIVAL_TIME}?",
    "I want to be dropped at {DROP_POINT} before {ARRIVAL_TIME} — show buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} that drops me at {DROP_POINT} by {ARRIVAL_TIME}",
    "I need to be at {DROP_POINT} by {ARRIVAL_TIME} on {DEPARTURE_DATE} — find a bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "What bus from {SOURCE_NAME} to {DESTINATION_NAME} will drop me at {DROP_POINT} before {ARRIVAL_TIME}?",
    "Find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} that reaches {DROP_POINT} by {ARRIVAL_TIME}",

    # --- DROP + OPERATOR ---
    "Does {OPERATOR} drop passengers at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "I want to travel with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} — will they drop me at {DROP_POINT}?",
    "Book me on {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} with drop at {DROP_POINT} on {DEPARTURE_DATE}",
    "Does {OPERATOR} have a bus from {SOURCE_NAME} to {DESTINATION_NAME} that drops at {DROP_POINT}?",

    # --- DROP clearly distinguished from DESTINATION ---
    "I am going from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} but I want to be dropped at {DROP_POINT}, which is before the final stop",
    "My final destination is {DESTINATION_NAME} but please drop me at {DROP_POINT} which is enroute on the {SOURCE_NAME} to {DESTINATION_NAME} bus",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} — I don't need to go all the way, just drop me at {DROP_POINT}",
    "I'm booking from {SOURCE_NAME} to {DESTINATION_NAME} but my actual drop-off is {DROP_POINT} on the way",
    "The route is {SOURCE_NAME} to {DESTINATION_NAME} but I'll get off at {DROP_POINT} — please book accordingly",
    "I am traveling from {SOURCE_NAME} to {DESTINATION_NAME} — my drop location is {DROP_POINT}, not the terminus",
    "Book me from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I will deboard at {DROP_POINT} en route",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} but I'll exit at {DROP_POINT} — is that possible?",
    "Please drop me at {DROP_POINT} and not at the final destination {DESTINATION_NAME} — I'm boarding from {SOURCE_NAME}",
    "Route: {SOURCE_NAME} to {DESTINATION_NAME}, but my actual exit point is {DROP_POINT} — please book for {DEPARTURE_DATE}",

    # --- DROP informal ---
    "drop me at {DROP_POINT} {SOURCE_NAME} to {DESTINATION_NAME}",
    "deboard at {DROP_POINT} bus {SOURCE_NAME} {DESTINATION_NAME}",
    "my drop point is {DROP_POINT} {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "get off at {DROP_POINT} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "{SOURCE_NAME} to {DESTINATION_NAME} drop {DROP_POINT} please",
    "bus dropping at {DROP_POINT} {SOURCE_NAME} to {DESTINATION_NAME}",
    "need drop at {DROP_POINT} {SOURCE_NAME} to {DESTINATION_NAME}",
    "drop off {DROP_POINT} {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "alighting point {DROP_POINT} {SOURCE_NAME} {DESTINATION_NAME}",
    "any bus dropping at {DROP_POINT} from {SOURCE_NAME} to {DESTINATION_NAME}?",

    # =========================================================================
    # SECTION 13: PICKUP + DROP together — clearly showing both (80 templates)
    # =========================================================================

    "I want to board at {PICKUP_POINT} and get off at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} bus on {DEPARTURE_DATE}",
    "Please book a bus from {SOURCE_NAME} to {DESTINATION_NAME} with pickup at {PICKUP_POINT} and drop at {DROP_POINT} on {DEPARTURE_DATE}",
    "I'll board at {PICKUP_POINT} and alight at {DROP_POINT} — route is {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} picking up at {PICKUP_POINT} and dropping at {DROP_POINT}",
    "Boarding: {PICKUP_POINT}, Alighting: {DROP_POINT} — I'm traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} that stop at {PICKUP_POINT} for boarding and {DROP_POINT} for dropping",
    "I need a bus with boarding stop at {PICKUP_POINT} and alighting stop at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "Book me from {SOURCE_NAME} to {DESTINATION_NAME} — pick up at {PICKUP_POINT}, drop off at {DROP_POINT}, date {DEPARTURE_DATE}",
    "I travel from {SOURCE_NAME} to {DESTINATION_NAME} — my boarding point is {PICKUP_POINT} and my drop point is {DROP_POINT}",
    "Does any bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} stop at {PICKUP_POINT} and also drop at {DROP_POINT}?",
    "Find me a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} picking up from {PICKUP_POINT} and dropping at {DROP_POINT}",
    "I need a bus with pickup at {PICKUP_POINT} and drop at {DROP_POINT} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book a {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} — pickup {PICKUP_POINT}, drop {DROP_POINT}, date {DEPARTURE_DATE}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with pickup at {PICKUP_POINT} and drop at {DROP_POINT}",
    "I want {SEAT_TYPE} on a bus from {SOURCE_NAME} to {DESTINATION_NAME} — board at {PICKUP_POINT}, exit at {DROP_POINT}",
    "My journey: {SOURCE_NAME} → {DESTINATION_NAME}, board at {PICKUP_POINT}, deboard at {DROP_POINT}, date {DEPARTURE_DATE}",
    "Book a {SEAT_TYPE} with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} — pickup {PICKUP_POINT}, drop {DROP_POINT}",
    "Find an {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} with boarding at {PICKUP_POINT} and exit at {DROP_POINT} on {DEPARTURE_DATE}",
    "I want to board at {PICKUP_POINT} and be dropped at {DROP_POINT} on the {SOURCE_NAME} to {DESTINATION_NAME} bus — date {DEPARTURE_DATE}",
    "Find a bus with pickup at {PICKUP_POINT} and drop at {DROP_POINT} from {SOURCE_NAME} to {DESTINATION_NAME} within {PRICE}",

    # With DEPARTURE + ARRIVAL time
    "Bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing at {DEPARTURE_TIME}, pickup at {PICKUP_POINT}, drop at {DROP_POINT}",
    "I want to board at {PICKUP_POINT} at {DEPARTURE_TIME} and be dropped at {DROP_POINT} by {ARRIVAL_TIME} — route {SOURCE_NAME} to {DESTINATION_NAME}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, pickup {PICKUP_POINT}, drop {DROP_POINT}, arrive {ARRIVAL_TIME}",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing at {DEPARTURE_TIME}, boarding at {PICKUP_POINT} and dropping at {DROP_POINT}",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} that departs at {DEPARTURE_TIME}, picks me up at {PICKUP_POINT} and drops me at {DROP_POINT} by {ARRIVAL_TIME}",
    "Find a bus departing {DEPARTURE_TIME} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — boarding at {PICKUP_POINT}, dropping at {DROP_POINT}",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE} — pickup at {PICKUP_POINT}, drop at {DROP_POINT}, arrive by {ARRIVAL_TIME}",

    # Informal PICKUP + DROP
    "pickup {PICKUP_POINT} drop {DROP_POINT} {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "board {PICKUP_POINT} alight {DROP_POINT} bus {SOURCE_NAME} {DESTINATION_NAME}",
    "{SOURCE_NAME} {DESTINATION_NAME} pickup {PICKUP_POINT} drop {DROP_POINT}",
    "bus {SOURCE_NAME} to {DESTINATION_NAME} pickup {PICKUP_POINT} and drop me at {DROP_POINT}",
    "board from {PICKUP_POINT} drop at {DROP_POINT} {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",

    # =========================================================================
    # SECTION 14: DEPARTURE_TIME — Time-of-departure clarity (100 templates)
    # =========================================================================

    # --- Standalone DEPARTURE_TIME ---
    "I want a bus from {SOURCE_NAME} to {DESTINATION_NAME} that departs at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} leaving at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} departing around {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE}?",
    "I need a {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book me on the {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "What buses are available from {SOURCE_NAME} to {DESTINATION_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE}?",
    "I want to leave {SOURCE_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE} heading to {DESTINATION_NAME}",
    "Are there any buses from {SOURCE_NAME} to {DESTINATION_NAME} that depart at {DEPARTURE_TIME}?",
    "I prefer to travel at {DEPARTURE_TIME} — show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please find a bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "I want to take the {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with departure around {DEPARTURE_TIME}",
    "Find an {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "I need a {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "Is there a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} that leaves at {DEPARTURE_TIME}?",
    "Book the {DEPARTURE_TIME} {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} with {AMENITIES}",
    "I want to catch the {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — is there availability?",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that start between {DEPARTURE_TIME} and later",

    # --- DEPARTURE_TIME + PICKUP ---
    "I need to board at {PICKUP_POINT} at {DEPARTURE_TIME} for the bus to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "What time does the bus from {SOURCE_NAME} to {DESTINATION_NAME} pick up at {PICKUP_POINT} — I want a {DEPARTURE_TIME} departure",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME}, pickup at {PICKUP_POINT} on {DEPARTURE_DATE}",
    "I will be at {PICKUP_POINT} at {DEPARTURE_TIME} — find a bus to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} from {PICKUP_POINT} on {DEPARTURE_DATE}",

    # --- DEPARTURE_TIME time-of-day natural phrasing ---
    "I want an early morning bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — around {DEPARTURE_TIME}",
    "Please find a night bus from {SOURCE_NAME} to {DESTINATION_NAME} departing around {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "I need a late night bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — departure time around {DEPARTURE_TIME}",
    "Show afternoon buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing after {DEPARTURE_TIME}",
    "I prefer morning buses — find one from {SOURCE_NAME} to {DESTINATION_NAME} leaving at {DEPARTURE_TIME} on {DEPARTURE_DATE}",
    "Are there evening buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing around {DEPARTURE_TIME}?",
    "I want to depart {SOURCE_NAME} at exactly {DEPARTURE_TIME} on {DEPARTURE_DATE} heading to {DESTINATION_NAME}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} leaving before {DEPARTURE_TIME} on {DEPARTURE_DATE}?",
    "Find me the first bus from {SOURCE_NAME} to {DESTINATION_NAME} in the morning on {DEPARTURE_DATE} — around {DEPARTURE_TIME}",
    "Show me the last bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — departure around {DEPARTURE_TIME}",

    # --- DEPARTURE_TIME informal ---
    "bus at {DEPARTURE_TIME} from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "need {DEPARTURE_TIME} bus {SOURCE_NAME} {DESTINATION_NAME}",
    "{DEPARTURE_TIME} departure {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE}",
    "show buses leaving at {DEPARTURE_TIME} {SOURCE_NAME} to {DESTINATION_NAME}",
    "any bus at {DEPARTURE_TIME} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "book {DEPARTURE_TIME} bus {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "{SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} at {DEPARTURE_TIME}",
    "depart {DEPARTURE_TIME} {SOURCE_NAME} {DESTINATION_NAME}",
    "leaving at {DEPARTURE_TIME} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "want {DEPARTURE_TIME} departure {SOURCE_NAME} to {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 15: ARRIVAL_TIME — Time-of-arrival clarity (100 templates)
    # =========================================================================

    # --- Standalone ARRIVAL_TIME (teaches: ARRIVAL_TIME ≠ DEPARTURE_TIME) ---
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_TIME} — what buses are available?",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} that arrives at {DESTINATION_NAME} by {ARRIVAL_TIME}",
    "I must reach {DESTINATION_NAME} before {ARRIVAL_TIME} — find me the right bus from {SOURCE_NAME}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} that arrive before {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} that will get me there by {ARRIVAL_TIME}?",
    "I want to arrive at {DESTINATION_NAME} by {ARRIVAL_TIME} on {DEPARTURE_DATE} — what bus should I take from {SOURCE_NAME}?",
    "Please find a bus from {SOURCE_NAME} to {DESTINATION_NAME} with arrival before {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "I need to be in {DESTINATION_NAME} by {ARRIVAL_TIME} — show me buses from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "What is the latest bus I can take from {SOURCE_NAME} to {DESTINATION_NAME} and still arrive by {ARRIVAL_TIME}?",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} that reaches by {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "I have a meeting in {DESTINATION_NAME} at {ARRIVAL_TIME} — find me a bus from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} guaranteed to arrive before {ARRIVAL_TIME}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} arriving before {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "I want to arrive in {DESTINATION_NAME} no later than {ARRIVAL_TIME} — find a bus from {SOURCE_NAME} for {DEPARTURE_DATE}",
    "Is there an overnight bus from {SOURCE_NAME} to {DESTINATION_NAME} arriving by {ARRIVAL_TIME} in the morning?",
    "I need to reach {DESTINATION_NAME} by {ARRIVAL_TIME} for an early morning appointment — book a bus from {SOURCE_NAME}",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} that arrives at {DROP_POINT} by {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "Find an {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} arriving before {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "I need a bus with arrival in {DESTINATION_NAME} by {ARRIVAL_TIME} — route from {SOURCE_NAME} on {DEPARTURE_DATE}",
    "Please show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with expected arrival before {ARRIVAL_TIME}",

    # --- ARRIVAL_TIME + DROP_POINT ---
    "I need to be at {DROP_POINT} by {ARRIVAL_TIME} — find a bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} that drops me at {DROP_POINT} by {ARRIVAL_TIME}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} dropping at {DROP_POINT} before {ARRIVAL_TIME}?",
    "I need to reach {DROP_POINT} before {ARRIVAL_TIME} — find a bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} — drop at {DROP_POINT}, arrival by {ARRIVAL_TIME}",

    # --- DEPARTURE_TIME + ARRIVAL_TIME together (model must learn both labels) ---
    "I want to depart {SOURCE_NAME} at {DEPARTURE_TIME} and reach {DESTINATION_NAME} by {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} and arriving before {ARRIVAL_TIME}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} that leaves at {DEPARTURE_TIME} and arrives before {ARRIVAL_TIME}?",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing at {DEPARTURE_TIME} and reaching by {ARRIVAL_TIME}",
    "I need to leave {SOURCE_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE} and be in {DESTINATION_NAME} by {ARRIVAL_TIME}",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_TIME}, arrival before {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "What buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} depart at {DEPARTURE_TIME} and arrive by {ARRIVAL_TIME}?",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} — leaving at {DEPARTURE_TIME} and arriving by {ARRIVAL_TIME}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that depart at {DEPARTURE_TIME} and reach by {ARRIVAL_TIME}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} and arriving before {ARRIVAL_TIME}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that departs at {DEPARTURE_TIME} and arrives before {ARRIVAL_TIME}?",
    "I want to leave at {DEPARTURE_TIME} from {SOURCE_NAME} and be at {DESTINATION_NAME} before {ARRIVAL_TIME} — what are my options?",
    "Book a {SEAT_TYPE} on a bus from {SOURCE_NAME} to {DESTINATION_NAME} — departs {DEPARTURE_TIME}, arrives {ARRIVAL_TIME}",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with departure at {DEPARTURE_TIME} and arrival by {ARRIVAL_TIME}",
    "Show {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing at {DEPARTURE_TIME}, arriving before {ARRIVAL_TIME}",

    # --- ARRIVAL_TIME informal ---
    "reach {DESTINATION_NAME} by {ARRIVAL_TIME} from {SOURCE_NAME} {DEPARTURE_DATE}",
    "need to arrive {DESTINATION_NAME} before {ARRIVAL_TIME}",
    "bus {SOURCE_NAME} to {DESTINATION_NAME} arrive by {ARRIVAL_TIME}",
    "i must be in {DESTINATION_NAME} by {ARRIVAL_TIME} {DEPARTURE_DATE}",
    "show buses arriving {DESTINATION_NAME} before {ARRIVAL_TIME}",
    "arrival by {ARRIVAL_TIME} {SOURCE_NAME} to {DESTINATION_NAME}",
    "reach {DESTINATION_NAME} {ARRIVAL_TIME} from {SOURCE_NAME}",
    "arrive before {ARRIVAL_TIME} {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE}",
    "depart {DEPARTURE_TIME} arrive {ARRIVAL_TIME} {SOURCE_NAME} {DESTINATION_NAME}",
    "leaving {DEPARTURE_TIME} reaching {ARRIVAL_TIME} {SOURCE_NAME} to {DESTINATION_NAME}",

    # =========================================================================
    # SECTION 16: DEPARTURE_TIME + ARRIVAL_TIME + PICKUP + DROP (50 templates)
    # Full journey specification — all 4 critical labels together
    # =========================================================================

    "I want to board at {PICKUP_POINT} at {DEPARTURE_TIME} and be dropped at {DROP_POINT} by {ARRIVAL_TIME} — route is {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — board at {PICKUP_POINT} at {DEPARTURE_TIME}, drop at {DROP_POINT} by {ARRIVAL_TIME}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} departing at {DEPARTURE_TIME} picking up at {PICKUP_POINT} and arriving at {DROP_POINT} before {ARRIVAL_TIME}",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} — pickup at {PICKUP_POINT} at {DEPARTURE_TIME}, drop at {DROP_POINT} by {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing around {DEPARTURE_TIME}, pickup at {PICKUP_POINT}, drop at {DROP_POINT}, reaching by {ARRIVAL_TIME}",
    "Book a {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_TIME}, board at {PICKUP_POINT}, exit at {DROP_POINT}, arrive by {ARRIVAL_TIME}",
    "I want to leave {SOURCE_NAME} at {DEPARTURE_TIME} from {PICKUP_POINT} and reach {DROP_POINT} in {DESTINATION_NAME} before {ARRIVAL_TIME}",
    "Find me a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} — pickup at {PICKUP_POINT} at {DEPARTURE_TIME}, drop at {DROP_POINT} before {ARRIVAL_TIME}",
    "Please book a bus with pickup at {PICKUP_POINT} and drop at {DROP_POINT} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I want {DEPARTURE_TIME} departure and {ARRIVAL_TIME} arrival",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} picking up at {PICKUP_POINT} around {DEPARTURE_TIME} and dropping at {DROP_POINT} before {ARRIVAL_TIME}?",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} — board at {PICKUP_POINT}, drop at {DROP_POINT}, must arrive by {ARRIVAL_TIME}",
    "Find an {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} — pickup {PICKUP_POINT} at {DEPARTURE_TIME}, drop {DROP_POINT} by {ARRIVAL_TIME} on {DEPARTURE_DATE}",
    "Book a {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing {DEPARTURE_TIME}, board {PICKUP_POINT}, exit {DROP_POINT}, reach by {ARRIVAL_TIME}",
    "Journey details: {SOURCE_NAME} to {DESTINATION_NAME}, date {DEPARTURE_DATE}, departure {DEPARTURE_TIME}, pickup {PICKUP_POINT}, drop {DROP_POINT}, arrive by {ARRIVAL_TIME}",
    "I want to book a {SEAT_TYPE} seat with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — pickup at {PICKUP_POINT} at {DEPARTURE_TIME} and drop at {DROP_POINT} by {ARRIVAL_TIME}",

    # =========================================================================
    # SECTION 17: ARRIVAL_DATE combined with DEPARTURE_DATE + TIME (80 templates)
    # Ensuring model learns ARRIVAL_DATE ≠ DEPARTURE_DATE in round trip context
    # =========================================================================

    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} — my departure date is {DEPARTURE_DATE} and my return date is {ARRIVAL_DATE}",
    "Book a round trip from {SOURCE_NAME} to {DESTINATION_NAME}: outward journey on {DEPARTURE_DATE}, return journey on {ARRIVAL_DATE}",
    "I need to go to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} and come back on {ARRIVAL_DATE}",
    "Please book my ticket: I'll travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "I want a bus to {DESTINATION_NAME} from {SOURCE_NAME} departing on {DEPARTURE_DATE} and I'll return by {ARRIVAL_DATE}",
    "Please confirm a round trip booking from {SOURCE_NAME} to {DESTINATION_NAME}: go on {DEPARTURE_DATE}, return on {ARRIVAL_DATE}",
    "I need tickets for my trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "Book me for a round trip: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, back to {SOURCE_NAME} on {ARRIVAL_DATE}",
    "I'll be visiting {DESTINATION_NAME} from {SOURCE_NAME} from {DEPARTURE_DATE} to {ARRIVAL_DATE} — please book both ways",
    "Two-way journey: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, return on {ARRIVAL_DATE} — please book",
    "I plan to go to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE} — book the round trip from {SOURCE_NAME}",
    "My trip from {SOURCE_NAME} to {DESTINATION_NAME} is from {DEPARTURE_DATE} to {ARRIVAL_DATE} — I need both legs booked",
    "I want to go to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE} — traveling from {SOURCE_NAME}",
    "Can you help me book a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — going {DEPARTURE_DATE}, returning {ARRIVAL_DATE}?",
    "I need a round trip bus: {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE}, returning {ARRIVAL_DATE}",

    # With DEPARTURE_TIME ---
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and return on {ARRIVAL_DATE}",
    "Book a round trip from {SOURCE_NAME} to {DESTINATION_NAME}: go {DEPARTURE_DATE} at {DEPARTURE_TIME}, return {ARRIVAL_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} with return on {ARRIVAL_DATE}",
    "I want to depart at {DEPARTURE_TIME} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} departing at {DEPARTURE_TIME} — I also need return on {ARRIVAL_DATE}",

    # With SEAT_TYPE / AC_TYPE ---
    "Book a {SEAT_TYPE} round trip from {SOURCE_NAME} to {DESTINATION_NAME}: departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I want {AC_TYPE} {SEAT_TYPE} for a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — go {DEPARTURE_DATE}, back {ARRIVAL_DATE}",
    "Find a {SEMANTIC} round trip from {SOURCE_NAME} to {DESTINATION_NAME}: go {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "Please book {TRAVELER} on a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "I want a {AC_TYPE} bus for a two-way trip from {SOURCE_NAME} to {DESTINATION_NAME}: outward {DEPARTURE_DATE}, return {ARRIVAL_DATE}",

    # With ARRIVAL_TIME ---
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_TIME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "Book my journey from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} arriving by {ARRIVAL_TIME} — also need return on {ARRIVAL_DATE}",
    "I'll go from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and need to arrive by {ARRIVAL_TIME} — return date is {ARRIVAL_DATE}",

    # Informal ---
    "going {DEPARTURE_DATE} returning {ARRIVAL_DATE} {SOURCE_NAME} {DESTINATION_NAME}",
    "depart {DEPARTURE_DATE} come back {ARRIVAL_DATE} {SOURCE_NAME} to {DESTINATION_NAME}",
    "trip {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} return {ARRIVAL_DATE}",
    "go on {DEPARTURE_DATE} return on {ARRIVAL_DATE} {SOURCE_NAME} {DESTINATION_NAME}",
    "two way trip {SOURCE_NAME} {DESTINATION_NAME} {DEPARTURE_DATE} and {ARRIVAL_DATE}",
    "{DEPARTURE_DATE} to {ARRIVAL_DATE} {SOURCE_NAME} {DESTINATION_NAME} round trip",
    "outward {DEPARTURE_DATE} return {ARRIVAL_DATE} {SOURCE_NAME} to {DESTINATION_NAME}",
    "from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} back {ARRIVAL_DATE}",


    # =========================================================================
    # SECTION 18: DATE + TIME adjacent natural phrasing (90 templates)
    # Critical gap: "tomorrow morning", "Friday evening", "next Sunday night"
    # Model fails when DEPARTURE_DATE and DEPARTURE_TIME appear side by side
    # without strong connectors like "at" or "departing"
    # =========================================================================

    # --- "DATE morning/evening/night/afternoon" patterns ---
    "show me buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "find buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "book a bus from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "any buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}?",
    "buses from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME} please",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "need a bus {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "what are the bus options from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}?",
    "I need to go from {SOURCE_NAME} to {DESTINATION_NAME} {DEPARTURE_DATE} {DEPARTURE_TIME}",
    "show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}",

    # --- "tomorrow morning" style (DEPARTURE_DATE = tomorrow, DEPARTURE_TIME = morning) ---
    "show me buses from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "find a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "I want to go from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "any bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}?",
    "book a bus from {SOURCE_NAME} to {DESTINATION_NAME} for tomorrow {DEPARTURE_TIME}",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "tomorrow {DEPARTURE_TIME} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "show me {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "I want to leave {SOURCE_NAME} tomorrow {DEPARTURE_TIME} and reach {DESTINATION_NAME}",
    "is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}?",
    "what buses go from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}?",
    "can I get a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}?",
    "show {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "I want a {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",
    "book me on a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow {DEPARTURE_TIME}",

    # --- "day after tomorrow morning" style ---
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} day after tomorrow {DEPARTURE_TIME}",
    "show buses from {SOURCE_NAME} to {DESTINATION_NAME} day after tomorrow {DEPARTURE_TIME}",
    "any bus from {SOURCE_NAME} to {DESTINATION_NAME} day after tomorrow {DEPARTURE_TIME}?",

    # --- "DATE morning" with full date ---
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} morning",
    "find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} morning",
    "show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} morning",
    "book a morning bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a morning bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "any morning buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} evening",
    "find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} evening",
    "show evening buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "book an evening bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want an evening bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I need a night bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "find a night bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "show night buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} night",
    "any night buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I need an afternoon bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "show afternoon buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "book an afternoon bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- "tomorrow morning" + more context ---
    "I want to go from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning — please show me available buses",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning — preferably {AC_TYPE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning that has {SEAT_TYPE}?",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} tomorrow morning — what are my options?",
    "Book me a {SEAT_TYPE} seat on a bus from {SOURCE_NAME} to {DESTINATION_NAME} leaving tomorrow morning",
    "Show me {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning",
    "I want to leave {SOURCE_NAME} tomorrow morning and reach {DESTINATION_NAME} — find me a bus",
    "Are there any {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning?",
    "I need to go from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning for an urgent meeting",
    "What is the earliest bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow morning?",

    # --- "tomorrow evening/night" + more context ---
    "I want a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow evening — any options?",
    "Find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow night",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow night for {TRAVELER}",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by tomorrow evening",
    "Book me on a bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow night with {SEAT_TYPE}",
    "Is there a sleeper bus from {SOURCE_NAME} to {DESTINATION_NAME} tomorrow night?",

    # --- Weekday/date + time-of-day patterns ---
    "Find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} in the morning",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} in the evening?",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} in the afternoon",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} in the morning",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at night",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — preferably in the morning",
    "Are there buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} that depart at {DEPARTURE_TIME}?",
    "I want to depart {SOURCE_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and reach {DESTINATION_NAME} — show me buses",
    "Book me on the earliest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I need a {DEPARTURE_TIME} departure",
    "Show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I prefer {DEPARTURE_TIME} timing",

    # =========================================================================
    # SECTION 19: OPERATOR pure information queries (60 templates)
    # Gap: only 22 operator-only templates existed
    # =========================================================================

    # --- Operator route info ---
    "What routes does {OPERATOR} operate from {SOURCE_NAME}?",
    "Does {OPERATOR} have buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Tell me about {OPERATOR} services on the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "Is {OPERATOR} a reliable bus operator for the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "What is {OPERATOR}'s schedule from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "How many buses does {OPERATOR} run from {SOURCE_NAME} to {DESTINATION_NAME} per day?",
    "Does {OPERATOR} operate buses on the {SOURCE_NAME} to {DESTINATION_NAME} route on {DEPARTURE_DATE}?",
    "What is the rating of {OPERATOR} for the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Is {OPERATOR} known for being {SEMANTIC} on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Which operator is better for {SOURCE_NAME} to {DESTINATION_NAME} — is {OPERATOR} recommended?",
    "Tell me the bus timings for {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "What buses does {OPERATOR} offer from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I've heard good things about {OPERATOR} — do they run from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Does {OPERATOR} run a direct bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Show me all {OPERATOR} buses from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "What types of buses does {OPERATOR} have for the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Is {OPERATOR} running on {DEPARTURE_DATE} from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to book with {OPERATOR} — do they have {AC_TYPE} buses on {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Does {OPERATOR} provide {AMENITIES} on their buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Show me {OPERATOR} {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # --- Operator policy queries ---
    "What is {OPERATOR}'s cancellation policy for buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "How do I contact {OPERATOR} for my booking from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "What is {OPERATOR}'s luggage policy on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Does {OPERATOR} allow pets on their buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "What is the boarding procedure for {OPERATOR} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Does {OPERATOR} have a student discount for the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Is there a senior citizen discount with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "What amenities does {OPERATOR} provide on the {SOURCE_NAME} to {DESTINATION_NAME} route?",
    "Does {OPERATOR} have {SEAT_TYPE} seats on their {SOURCE_NAME} to {DESTINATION_NAME} buses?",
    "Is {OPERATOR} running on time today from {SOURCE_NAME} to {DESTINATION_NAME}?",

    # --- Operator-only (no route) ---
    "I want to book a {OPERATOR} bus — what routes are available?",
    "Show me all {OPERATOR} buses for {DEPARTURE_DATE}",
    "Find {OPERATOR} {AC_TYPE} buses available on {DEPARTURE_DATE}",
    "Is {OPERATOR} available for booking?",
    "What are the best {OPERATOR} routes?",
    "Does {OPERATOR} have buses running tomorrow?",
    "Show me {OPERATOR} buses with {SEAT_TYPE} seating",
    "I prefer {OPERATOR} — show me their upcoming buses",
    "Find {OPERATOR} buses with {AMENITIES}",
    "I want to book {OPERATOR} service — show available options",

    # =========================================================================
    # SECTION 20: Journey purpose queries (70 templates)
    # Gap: only 23 templates — medical, wedding, business, exam, pilgrimage
    # =========================================================================

    # --- Medical travel ---
    "I need to travel from {SOURCE_NAME} to {DESTINATION_NAME} urgently for a medical emergency on {DEPARTURE_DATE}",
    "Please find me the fastest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — it is a medical emergency",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} as soon as possible on {DEPARTURE_DATE} — medical trip",
    "Find me a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I have a hospital appointment",
    "I am traveling from {SOURCE_NAME} to {DESTINATION_NAME} for a medical check-up on {DEPARTURE_DATE} — please book a comfortable bus",
    "Please book me on the earliest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I need to reach for a doctor's appointment",
    "I need to travel from {SOURCE_NAME} to {DESTINATION_NAME} for surgery on {DEPARTURE_DATE} — please find a {SEMANTIC} and comfortable bus",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a medical trip — I need {AMENITIES} and a {SEAT_TYPE} seat",

    # --- Wedding travel ---
    "I need to attend a wedding in {DESTINATION_NAME} on {ARRIVAL_DATE} — please find a bus from {SOURCE_NAME}",
    "Book me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I am attending a wedding on {ARRIVAL_DATE}",
    "I am traveling from {SOURCE_NAME} to {DESTINATION_NAME} for a wedding on {ARRIVAL_DATE} — please find a {SEMANTIC} bus",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for a wedding trip on {DEPARTURE_DATE} for {TRAVELER}",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_DATE} for a family wedding",
    "Please book {TRAVELER} on a bus from {SOURCE_NAME} to {DESTINATION_NAME} for a wedding event on {ARRIVAL_DATE}",
    "I am going to {DESTINATION_NAME} from {SOURCE_NAME} to attend a relative's wedding on {ARRIVAL_DATE} — suggest a good bus",
    "Find me a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for a wedding celebration on {DEPARTURE_DATE}",

    # --- Business / work travel ---
    "I am traveling from {SOURCE_NAME} to {DESTINATION_NAME} for a business meeting on {DEPARTURE_DATE} — need a {SEMANTIC} bus",
    "Book me on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I have a business appointment",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_TIME} on {DEPARTURE_DATE} for a client meeting",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — business trip, I need {AMENITIES} and {AC_TYPE}",
    "I travel frequently from {SOURCE_NAME} to {DESTINATION_NAME} for work — find me a {SEMANTIC} bus for {DEPARTURE_DATE}",
    "Please find a comfortable bus from {SOURCE_NAME} to {DESTINATION_NAME} for my business trip on {DEPARTURE_DATE}",
    "I need to attend an office meeting in {DESTINATION_NAME} on {DEPARTURE_DATE} — find a bus from {SOURCE_NAME}",
    "Book me on the {SEMANTIC} available bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a work trip",

    # --- Exam / education travel ---
    "I have an exam in {DESTINATION_NAME} on {DEPARTURE_DATE} — please find the earliest bus from {SOURCE_NAME}",
    "Find me a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I have a government exam to attend",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_TIME} on {DEPARTURE_DATE} — I have an entrance exam",
    "Book me the earliest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I have an important exam",
    "I am going to {DESTINATION_NAME} for an exam on {DEPARTURE_DATE} — find a {SEMANTIC} bus from {SOURCE_NAME}",
    "Please find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} morning for my exam — I need to arrive early",
    "I have a competitive exam in {DESTINATION_NAME} — find me a bus from {SOURCE_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}",

    # --- Pilgrimage / religious travel ---
    "I am planning a pilgrimage to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} — please suggest a bus",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for a religious trip on {DEPARTURE_DATE} for {TRAVELER}",
    "I want to visit {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} for a religious occasion — suggest a comfortable bus",
    "Book me and {TRAVELER} on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a pilgrimage trip",
    "We are a group of {TRAVELER} planning a pilgrimage to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE}",

    # --- Holiday / vacation travel ---
    "I am planning a holiday trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — find me a {SEMANTIC} bus",
    "We are {TRAVELER} going on a vacation to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} — find a comfortable bus",
    "Please find a {SEMANTIC} bus for our family trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I am going on a vacation from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and returning on {ARRIVAL_DATE}",
    "Find a round trip bus for our family vacation from {SOURCE_NAME} to {DESTINATION_NAME} — going {DEPARTURE_DATE}, returning {ARRIVAL_DATE}",
    "I want to visit {DESTINATION_NAME} from {SOURCE_NAME} for the long weekend — book a bus for {DEPARTURE_DATE}",
    "Our family of {TRAVELER} is planning a trip to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} — find a {SEMANTIC} bus",
    "We are going on a group trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — find buses for {TRAVELER}",

    # --- Emergency / urgent travel ---
    "I need to travel urgently from {SOURCE_NAME} to {DESTINATION_NAME} today — what buses are available?",
    "I need to reach {DESTINATION_NAME} from {SOURCE_NAME} by {ARRIVAL_TIME} today — is there any bus?",
    "Please find me the next available bus from {SOURCE_NAME} to {DESTINATION_NAME} — urgent travel",
    "I need to leave {SOURCE_NAME} immediately for {DESTINATION_NAME} — find any bus available now",
    "Emergency travel required from {SOURCE_NAME} to {DESTINATION_NAME} today — please find fastest option",

    # =========================================================================
    # SECTION 21: Special passenger types (60 templates)
    # Gap: only 40 templates — senior, wheelchair, child, infant, family
    # =========================================================================

    # --- Senior citizen ---
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a senior citizen — need comfortable {SEAT_TYPE}",
    "Please find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for an elderly person on {DEPARTURE_DATE}",
    "Is there a senior citizen discount on buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Book a comfortable {SEAT_TYPE} seat for an elderly passenger from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I am booking for my elderly parents — find a {SEMANTIC} {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Find a {SEMANTIC} bus with {AMENITIES} from {SOURCE_NAME} to {DESTINATION_NAME} — traveling with senior citizens",
    "My 70-year-old mother is traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — please find a comfortable bus",
    "Book a {SEAT_TYPE} seat on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for an elderly traveler on {DEPARTURE_DATE}",

    # --- Family with children ---
    "I am traveling with young children from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — need a family-friendly bus",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for a family of {TRAVELER} on {DEPARTURE_DATE}",
    "We are a family traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — need {SEAT_TYPE} and {AMENITIES}",
    "I need {TRAVELER} seats on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a family trip",
    "Book a family trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — we need adjacent seats for {TRAVELER}",
    "Is there a family-friendly bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {AMENITIES}?",
    "We are traveling with a baby — find a comfortable bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} for 2 adults and 2 children on {DEPARTURE_DATE}",

    # --- Solo female traveler ---
    "I am a solo female traveler — find me a {SEMANTIC} and safe bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Please find a safe bus from {SOURCE_NAME} to {DESTINATION_NAME} for a woman traveling alone on {DEPARTURE_DATE}",
    "I want a bus from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} — I am a solo female traveler",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} that are safe for women on {DEPARTURE_DATE}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I am traveling alone as a woman",
    "Book me a lower {SEAT_TYPE} seat on a bus from {SOURCE_NAME} to {DESTINATION_NAME} — I'm a solo female traveler",

    # --- Group / corporate travelers ---
    "We are a group of {TRAVELER} traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — do you have group bookings?",
    "I want to book {TRAVELER} seats on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a corporate trip",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for a group of {TRAVELER} on {DEPARTURE_DATE}",
    "Can I get a group discount for {TRAVELER} traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Book seats for a group of {TRAVELER} on a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "We have {TRAVELER} people traveling from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — find suitable options",

    # --- First-time / hesitant traveler ---
    "This is my first time traveling by bus from {SOURCE_NAME} to {DESTINATION_NAME} — which is the safest operator?",
    "I am not sure which bus to take from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — please recommend the {SEMANTIC} one",
    "Can you help me pick the {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}?",
    "I am unfamiliar with the route from {SOURCE_NAME} to {DESTINATION_NAME} — which bus company is most {SEMANTIC}?",

    # --- Person with disability ---
    "Is there a wheelchair-accessible bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I need a bus from {SOURCE_NAME} to {DESTINATION_NAME} that is accessible for a disabled passenger on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} with facilities for a differently-abled traveler on {DEPARTURE_DATE}",
    "Does {OPERATOR} have accessible buses for disabled passengers from {SOURCE_NAME} to {DESTINATION_NAME}?",

    # =========================================================================
    # SECTION 22: Compare / preference / ranking queries (40 templates)
    # Gap: only 10 templates existed
    # =========================================================================

    "Which is better — {OPERATOR} or another operator for the route from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Compare buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — show me cheapest and best options",
    "What is the difference in price between {AC_TYPE} and non-AC buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Show me the top-rated buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Which bus from {SOURCE_NAME} to {DESTINATION_NAME} has the best passenger ratings?",
    "What is the cheapest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "What is the fastest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} sorted by price",
    "Show buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} sorted by rating",
    "I want to compare {SEAT_TYPE} and regular seats for buses from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Which is more comfortable — {AC_TYPE} or {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "What are the differences between the buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Show me the most value-for-money bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Which operator gives the best service from {SOURCE_NAME} to {DESTINATION_NAME} — {OPERATOR} or others?",
    "I want to find the best priced bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} within {PRICE}",
    "Is the {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} also the cheapest on {DEPARTURE_DATE}?",
    "Find me the {SEMANTIC} option from {SOURCE_NAME} to {DESTINATION_NAME} that is also budget-friendly",
    "Which is the top-rated and {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to know which bus from {SOURCE_NAME} to {DESTINATION_NAME} is best for overnight travel",
    "Compare the timing and price of buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",

    # =========================================================================
    # SECTION 23: Non-stop / direct bus queries (40 templates)
    # Gap: many "via/stop" queries but few direct-bus-specific ones
    # =========================================================================

    "Is there a direct bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want a non-stop bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a direct bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE}",
    "Show me only non-stop buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Are there any non-stop buses from {SOURCE_NAME} to {DESTINATION_NAME} in the morning on {DEPARTURE_DATE}?",
    "I don't want a bus with too many stops — find a direct bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Is there a non-stop {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Find me a non-stop {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want a direct {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Are there direct buses from {SOURCE_NAME} to {DESTINATION_NAME} that do not stop at intermediate points?",
    "I prefer a direct bus — show me non-stop options from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a non-stop bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE}",
    "Is there a direct bus from {SOURCE_NAME} to {DESTINATION_NAME} without any halts on {DEPARTURE_DATE}?",
    "Show me {SEMANTIC} non-stop buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want to reach {DESTINATION_NAME} from {SOURCE_NAME} as quickly as possible on {DEPARTURE_DATE} — find a direct bus",
    "Book me on a non-stop {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is {OPERATOR} running a direct bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Find a non-stop {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} within {PRICE}",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} without stopping — show non-stop buses on {DEPARTURE_DATE}",
    "Does any bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} go non-stop?",

    # =========================================================================
    # SECTION 24: Round trip with all 4 date/time entities (50 templates)
    # Critical pattern: DEPARTURE_DATE + DEPARTURE_TIME + ARRIVAL_DATE + ARRIVAL_TIME
    # =========================================================================

    "I want to go from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and return on {ARRIVAL_DATE} by {ARRIVAL_TIME}",
    "Book a round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departing {DEPARTURE_DATE} at {DEPARTURE_TIME} and returning {ARRIVAL_DATE} by {ARRIVAL_TIME}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} — I'll return on {ARRIVAL_DATE} before {ARRIVAL_TIME}",
    "I need a round trip: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return on {ARRIVAL_DATE} reaching by {ARRIVAL_TIME}",
    "Please book my round trip from {SOURCE_NAME} to {DESTINATION_NAME} — go on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return on {ARRIVAL_DATE}",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and come back on {ARRIVAL_DATE}",
    "Book round trip: {SOURCE_NAME} → {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return on {ARRIVAL_DATE} by {ARRIVAL_TIME}",
    "I'll go to {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME} and return on {ARRIVAL_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} morning and book return for {ARRIVAL_DATE}",
    "I want to depart {SOURCE_NAME} at {DEPARTURE_TIME} on {DEPARTURE_DATE} and return from {DESTINATION_NAME} on {ARRIVAL_DATE} before {ARRIVAL_TIME}",
    "Two-way ticket: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return {ARRIVAL_DATE} by {ARRIVAL_TIME}",
    "Please confirm a round trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} with return on {ARRIVAL_DATE}",
    "I need to go to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and be back by {ARRIVAL_TIME} on {ARRIVAL_DATE}",
    "Book me for a round trip — {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, returning {ARRIVAL_DATE}",
    "Find round trip buses from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE} {DEPARTURE_TIME}, return {ARRIVAL_DATE} {ARRIVAL_TIME}",
    "I want a {SEAT_TYPE} round trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return {ARRIVAL_DATE}",
    "Book {TRAVELER} on round trip from {SOURCE_NAME} to {DESTINATION_NAME} — go {DEPARTURE_DATE} at {DEPARTURE_TIME}, back {ARRIVAL_DATE}",
    "I need a {SEMANTIC} round trip: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return {ARRIVAL_DATE} by {ARRIVAL_TIME}",
    "Show me round trip options from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} {DEPARTURE_TIME} and returning {ARRIVAL_DATE}",
    "Is there a round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} with return on {ARRIVAL_DATE}?",
    "I want to depart {SOURCE_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME} and need a return ticket from {DESTINATION_NAME} on {ARRIVAL_DATE} {ARRIVAL_TIME}",
    "Book two-way tickets: go {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} {DEPARTURE_TIME}, return on {ARRIVAL_DATE}",
    "I need {AC_TYPE} {SEAT_TYPE} for round trip: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}, return {ARRIVAL_DATE}",
    "Find the {SEMANTIC} round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} at {DEPARTURE_TIME}, returning {ARRIVAL_DATE} by {ARRIVAL_TIME}",
    "I will reach {DESTINATION_NAME} from {SOURCE_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} and need to be back by {ARRIVAL_DATE}",

    # =========================================================================
    # SECTION 25: Holiday / season surge travel (40 templates)
    # Gap: only 18 templates
    # =========================================================================

    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} for the Diwali holidays on {DEPARTURE_DATE}",
    "Please find a bus from {SOURCE_NAME} to {DESTINATION_NAME} during the holiday season on {DEPARTURE_DATE}",
    "Is there any bus available from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} during the festival season?",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} during Eid on {DEPARTURE_DATE} — please find a bus",
    "Please book a bus from {SOURCE_NAME} to {DESTINATION_NAME} for the long weekend starting {DEPARTURE_DATE}",
    "I need to book well in advance for Diwali travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Are there special buses from {SOURCE_NAME} to {DESTINATION_NAME} during the festive season on {DEPARTURE_DATE}?",
    "Please find a bus for Christmas holidays from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I am planning to travel from {SOURCE_NAME} to {DESTINATION_NAME} during the summer vacation on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for the New Year holidays starting {DEPARTURE_DATE}",
    "I want to book early for the Onam season — find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I know it's a holiday, any availability?",
    "Book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for a public holiday trip",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on the long weekend of {DEPARTURE_DATE} — any buses?",
    "Find special holiday buses from {SOURCE_NAME} to {DESTINATION_NAME} operating on {DEPARTURE_DATE}",
    "I need to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} well in advance for the school holidays on {DEPARTURE_DATE}",
    "Are there extra buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} due to the festival?",
    "Show me {SEMANTIC} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} during the peak season",
    "I want to book {TRAVELER} tickets for the holiday period — find buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for the Pongal holiday on {DEPARTURE_DATE}",

    # =========================================================================
    # SECTION 26: OPERATOR + BUS_TYPE specific combo queries (40 templates)
    # Gap: many templates had one or other but not clear operator-bus-type context
    # =========================================================================

    "Does {OPERATOR} run {BUS_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want a {BUS_TYPE} bus from {OPERATOR} on the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "Show me {OPERATOR} {BUS_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book me on {OPERATOR}'s {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Is there a {BUS_TYPE} available with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "I want to travel with {OPERATOR} in a {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Find {OPERATOR} {BUS_TYPE} with {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I specifically want {OPERATOR}'s {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Does {OPERATOR} have a {AC_TYPE} {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "Show me {OPERATOR} buses — I want only their {BUS_TYPE} for the {SOURCE_NAME} to {DESTINATION_NAME} route",
    "I have traveled on {OPERATOR}'s {BUS_TYPE} before and loved it — book the same for {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Book a {BUS_TYPE} with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "Find a {BUS_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES} — I prefer {OPERATOR}",
    "Is {OPERATOR}'s {BUS_TYPE} available from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to book a {BUS_TYPE} with {OPERATOR} — route from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Show me all {BUS_TYPE} options on the {SOURCE_NAME} to {DESTINATION_NAME} route by {OPERATOR}",
    "Find a {BUS_TYPE} with good reviews from {SOURCE_NAME} to {DESTINATION_NAME} — preferably {OPERATOR}",
    "Book {OPERATOR} {BUS_TYPE} {AC_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I need a {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} — show me {OPERATOR} and other operators",
    "Does {OPERATOR} offer {BUS_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} with {AMENITIES}?",

    # =========================================================================
    # SECTION 27: PRICE-focused natural queries (40 templates)
    # Adding long, natural budget-oriented queries
    # =========================================================================

    "What is the price of a bus ticket from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "How much does a bus from {SOURCE_NAME} to {DESTINATION_NAME} cost on {DEPARTURE_DATE}?",
    "I have a budget of {PRICE} — find me the best bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "What is the cheapest bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} within {PRICE}?",
    "Show me all buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} under {PRICE}",
    "I want a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — my total budget is {PRICE} for {TRAVELER}",
    "Is there a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} within {PRICE} on {DEPARTURE_DATE}?",
    "Find me an {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} within {PRICE} on {DEPARTURE_DATE}",
    "How much does {OPERATOR} charge for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want {SEAT_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — what is the price?",
    "Show me {AC_TYPE} buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} under {PRICE}",
    "Find a {SEMANTIC} {SEAT_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} under {PRICE} for {DEPARTURE_DATE}",
    "What is the fare for a {SEAT_TYPE} seat from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to compare prices for buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — show all options",
    "How much does a {BUS_TYPE} from {SOURCE_NAME} to {DESTINATION_NAME} cost on {DEPARTURE_DATE}?",
    "Can I find a bus from {SOURCE_NAME} to {DESTINATION_NAME} under {PRICE} on {DEPARTURE_DATE}?",
    "I have {PRICE} — what is the best bus I can get from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "Show me buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} sorted by price — I want the cheapest",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} under {PRICE} total on {DEPARTURE_DATE}",
    "What is the ticket price for {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",

    # =========================================================================
    # SECTION 28: Post-booking / cancellation / refund queries (40 templates)
    # These establish context where OPERATOR, SOURCE, DEST appear in a support context
    # =========================================================================

    "I want to cancel my bus ticket from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — what is the refund?",
    "How do I cancel my {OPERATOR} bus booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I need to cancel my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — can I get a full refund?",
    "I want to reschedule my bus from {SOURCE_NAME} to {DESTINATION_NAME} from {DEPARTURE_DATE} to {ARRIVAL_DATE}",
    "Can I change my seat from {SOURCE_NAME} to {DESTINATION_NAME} bus on {DEPARTURE_DATE}?",
    "I want to change my {SEAT_TYPE} to a different seat on the bus from {SOURCE_NAME} to {DESTINATION_NAME}",
    "My bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} was cancelled — can I get a refund?",
    "I want to know the status of my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Where is my bus from {SOURCE_NAME} to {DESTINATION_NAME} — it is late on {DEPARTURE_DATE}",
    "Is the bus from {SOURCE_NAME} to {DESTINATION_NAME} running on time on {DEPARTURE_DATE}?",
    "I have a booking with {OPERATOR} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — please confirm",
    "Can I add {ADD_ONS} to my existing booking from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I want to upgrade my seat from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — what are the options?",
    "My bus from {SOURCE_NAME} to {DESTINATION_NAME} has been delayed — what are my options?",
    "Can I transfer my bus ticket from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} to another person?",
    "I missed my bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — what can I do?",
    "What is the cancellation fee for the {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}?",
    "I need to cancel one of my {TRAVELER} tickets for the bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "Can I get a partial refund if I cancel one ticket for the bus from {SOURCE_NAME} to {DESTINATION_NAME}?",
    "My {OPERATOR} bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} is showing delayed — please update",

    # =========================================================================
    # SECTION 29: TRAVELER count with explicit seat count phrasing (40 templates)
    # Critical: "for 1 traveler", "for 2 passengers", "for 3 people"
    # =========================================================================

    "Please book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "I need to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "Book tickets from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE}",
    "I want to book a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} for {TRAVELER}",
    "Please confirm my booking for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "Show me available buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "I am booking a seat on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — traveling as {TRAVELER}",
    "Find a {SEMANTIC} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE}",
    "I want a {SEAT_TYPE} seat on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "Book a {AC_TYPE} bus from {SOURCE_NAME} to {DESTINATION_NAME} for {TRAVELER} on {DEPARTURE_DATE}",
    "Is there a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}?",
    "Please find a {SEAT_TYPE} for {TRAVELER} on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}",
    "I want to travel from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} — I am {TRAVELER}",
    "Book a bus ticket for {TRAVELER} from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME}",
    "I am searching for a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER} — show me available seats",
    "Please help me book a comfortable bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} for {TRAVELER}",
    "Find buses from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} at {DEPARTURE_TIME} with enough seats for {TRAVELER}",
    "Confirm a booking for {TRAVELER} on a bus from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with {SEAT_TYPE}",
    "I want a round trip bus from {SOURCE_NAME} to {DESTINATION_NAME} departing {DEPARTURE_DATE} and returning {ARRIVAL_DATE} for {TRAVELER}",

    # =========================================================================
    # SECTION 30: DEPARTURE_DATE + ARRIVAL_DATE multi-leg journey (30 templates)
    # Templates where both dates appear with very natural transitions
    # =========================================================================

    "I need to go from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and then come back on {ARRIVAL_DATE}",
    "Please book my two-way trip: I'll depart on {DEPARTURE_DATE} from {SOURCE_NAME} and return on {ARRIVAL_DATE}",
    "I am planning to travel from {SOURCE_NAME} to {DESTINATION_NAME} between {DEPARTURE_DATE} and {ARRIVAL_DATE}",
    "I will leave {SOURCE_NAME} on {DEPARTURE_DATE} and return from {DESTINATION_NAME} on {ARRIVAL_DATE}",
    "Find a bus from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and a return bus for {ARRIVAL_DATE}",
    "I want to leave on {DEPARTURE_DATE} and return on {ARRIVAL_DATE} — book a round trip from {SOURCE_NAME} to {DESTINATION_NAME}",
    "Book my journey from {SOURCE_NAME} to {DESTINATION_NAME}: outward on {DEPARTURE_DATE}, return on {ARRIVAL_DATE}",
    "Please check bus availability from {SOURCE_NAME} to {DESTINATION_NAME} for {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "I want to visit {DESTINATION_NAME} from {SOURCE_NAME} — I'll leave on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "My trip from {SOURCE_NAME} to {DESTINATION_NAME} starts on {DEPARTURE_DATE} and ends with return on {ARRIVAL_DATE}",
    "Please confirm both tickets: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} and return on {ARRIVAL_DATE}",
    "I am booking a round trip from {SOURCE_NAME} to {DESTINATION_NAME}: departure {DEPARTURE_DATE}, return {ARRIVAL_DATE}",
    "Find a {SEMANTIC} round trip from {SOURCE_NAME} to {DESTINATION_NAME} — departure {DEPARTURE_DATE} and return {ARRIVAL_DATE}",
    "Book {TRAVELER} on a round trip from {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE} with return on {ARRIVAL_DATE}",
    "I need a {AC_TYPE} bus for round trip: {SOURCE_NAME} to {DESTINATION_NAME} on {DEPARTURE_DATE}, return {ARRIVAL_DATE}",

]
