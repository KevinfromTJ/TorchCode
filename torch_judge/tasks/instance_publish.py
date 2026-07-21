"""Instance State Publish Operation — Pinduoduo 2026-07-19 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "Instance State Publish Operation",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "A single range-assign turned A into B. Let L be the first index where A and B differ "
        "and R the last. The target value must be V = B[L], and the operation must cover [L, R]. "
        "The operation is unique ONLY IF it cannot be extended: it fails to be unique when "
        "(L>0 and B[L-1]==V) or (R<n-1 and B[R+1]==V). Return '{L+1} {R+1} {V}' (1-indexed) or "
        "'-1'. Inputs guarantee at least one valid operation exists (so A != B)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.find_operation("00000", "01110") == "2 4 1", sol.find_operation("00000", "01110")
assert sol.find_operation("01000", "01110") == "-1"
assert sol.find_operation("111111", "100001") == "2 5 0"
assert sol.find_operation("0001000", "0111110") == "2 6 1"
assert sol.find_operation("0", "1") == "1 1 1"
assert sol.find_operation("00010", "01110") == "-1"
""",
        },
        {
            "name": "Unique middle operation",
            "code": """
sol = {fn}()
# only one contiguous block differs and neither neighbor equals the target
assert sol.find_operation("0000", "0110") == "2 3 1"
assert sol.find_operation("111", "101") == "2 2 0"
""",
        },
        {
            "name": "Left-extendable is not unique",
            "code": """
sol = {fn}()
# B[L-1] already equals target V=1 -> can extend left -> -1
assert sol.find_operation("1000", "1100") == "-1"
""",
        },
        {
            "name": "Right-extendable is not unique",
            "code": """
sol = {fn}()
# B[R+1] already equals target V=1 -> can extend right -> -1
assert sol.find_operation("0001", "0011") == "-1"
""",
        },
        {
            "name": "Both-extendable is not unique",
            "code": """
sol = {fn}()
# target block surrounded by same value on both sides
assert sol.find_operation("10001", "11011") == "-1"
""",
        },
        {
            "name": "Whole-string and boundary operations",
            "code": """
sol = {fn}()
# full-range operation, unique because both ends are at the boundary
assert sol.find_operation("000", "111") == "1 3 1"
# operation anchored at the left boundary
assert sol.find_operation("110", "010") == "1 1 0"
# operation anchored at the right boundary
assert sol.find_operation("011", "010") == "3 3 0"
""",
        },
        {
            "name": "Target value V=0 branch",
            "code": """
sol = {fn}()
assert sol.find_operation("1111", "1001") == "2 3 0"
assert sol.find_operation("0110", "0010") == "-1"   # left neighbor B[0]=0 == V
""",
        },
        {
            "name": "Range may contain already-target cells",
            "code": """
sol = {fn}()
# index 3 is already '1' in A but lies inside [L,R]; still a single op
assert sol.find_operation("0001000", "0111110") == "2 6 1"
# a run with an internal matching cell, unique
assert sol.find_operation("00100", "01110") == "2 4 1"
""",
        },
    ],
}
