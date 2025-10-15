
# Video 1: Python Basics for AI – Variables, Data Types, and Control Structures

# This video has its focus on:
# 1. Creating variables.
# 2. Creating and indexing lists.
# 3. Creating and accessing dictionaries.
# 4. Using if, elif, and else statements for control flow.
# 5. Looping: While loops and for loops.
# 6. Creating and calling a function.

# A. Variables in Python:
# In Python, you don't need to declare types; they are inferred automatically.

threat_score = 95           # Integer: Used for counting and scoring.
is_critical = True          # Boolean: Used for True/False logic.
source_ip = "192.168.1.10"  # String: Used for text and identifiers.
float_num = 3.14159         # Float: A number with a decimal point.
c_num = 1 + 2j              # Complex Number: Numbers with a real and imaginary part.

print(f"System IP: {source_ip}, Score: {threat_score}")
print("-" * 30)

# B. Lists
# Lists are ordered collections, perfect for sequences of events or feature sets.
log_events = ["Login Failed", "DB Access", "External Transfer", "Login Failed", "System Crash"]
print(f"Total events in log: {len(log_events)}")

# Accessing an element (indexing starts at 0)
print(f"First event: {log_events[0]}")
print(f"Last event: {log_events[-1]}")
print("-" * 30)

# C. Dictionaries (Key-Value Pairs) 
# Dictionaries are used for organized, labeled data, like mapping an IP 
# to its status.
user_profile = {
    "user_id": "grant_k",
    "role": "Admin",
    "status": "Active",
    "last_login": "2025-09-20"
}

print(f"User Role: {user_profile['role']}")
print("-" * 30)

# E. Control Flow: Conditional Logic (if/elif/else) 
# Essential for implementing security rules and decision-making.
if threat_score > 90 and is_critical:
    print("ALERT: Immediate action required (Threat Score is Critical).")
elif threat_score > 50:
    print("WARNING: Investigate user activity.")
else:
    print("Status: Normal.")
print("-" * 30)

# F. Control Flow: Iteration, for loops and while loops.
# Essential for processing large amounts of log entries and data batches.
print("Processing Events:")
suspicious_count = 0
for event in log_events:
    if "Failed" in event or "Crash" in event:
        print(f"    - Found a suspicious event: {event}")
        suspicious_count += 1
print(f"Total suspicious events found: {suspicious_count}")
print("-" * 30)

# A while loop continues executing its block of code AS LONG AS its condition is True.
while tasks_pending > 0:
	print(f"	- Processing Task #{task_number}. Remaining: {tasks_pending - 1}")
	# The condition variable must be changed *inside* the loop to eventually stop it.
	tasks_pending -= 1 # Decrement the counter to move toward the stopping condition (tasks_pending > 0 is False)
	task_number += 1

print("All pending tasks processed.")

# G. Defining a Reusable Function 
# Functions group logic into one reusable block, making code clean and scalable.
def check_authentication_status(username, attempts):
    """Returns a security recommendation based on login attempts.
       This type of comment is called a "doctsring", and you can use 
       a docstring to briefly explain what the function is 
       doing."""
    if attempts > 3:
        return f"User {username}: Too many failures. ACCOUNT LOCKOUT RECOMMENDED."
    else:
        return f"User {username}: Status OK."

# Calling the function
print(check_authentication_status("test_user_1", 5))
print(check_authentication_status("authorized_user", 2))
